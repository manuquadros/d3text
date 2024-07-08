import os

import datasets
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Model(torch.nn.Module):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.base_model = transformers.AutoModel.from_pretrained(model_id)

    def load_dataset(self, dataset: datasets.DatasetDict) -> None:
        """
        Load dataset and tokenize it, keeping track of NERC tags.
        """

        # We need to know the maximum length of a tokenized sequence for padding.
        max_length = max(
            len(sample["input_ids"])
            for split in dataset.map(
                lambda s: self._tokenizer(s["tokens"], is_split_into_words=True)
            ).values()
            for sample in split
        )

        data = dataset.map(
            lambda sample: self.tokenize_and_align(sample, max_length),
            remove_columns="tokens",
        )

        # One-hot encoding for the NERC labels
        self._label_encoder = sklearn.preprocessing.LabelEncoder()
        self._label_encoder.fit(
            [label
             for split in dataset.values()
             for sample in split
             for label in sample["nerc_tags"]] + ["#"]
        )

        self.num_labels = len(self._label_encoder.classes_)

        data = data.map(
            lambda sample: {
                "nerc_tags": torch.nn.functional.one_hot(
                    torch.tensor(self._label_encoder.transform(sample["nerc_tags"])),
                    num_classes=self.num_labels,
                ).to(torch.float16)
            }
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._data = data.with_format("torch", device=device)

    def tokenize_and_align(
        self, sample: dict[str, list[str]], max_length: int
    ) -> dict[str, list[str]]:
        sentence = self._tokenizer(
            sample["tokens"],
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length,
        )

        labels = []
        for idx in sentence.word_ids():
            if idx is None:
                labels.append("#")
            else:
                labels.append(sample["nerc_tags"][idx])

        return {"sentence": sentence, "nerc_tags": labels}

    def train_model(
        self,
        batch_size: int = 64,
        num_epochs: int = 5,
        shuffle: bool = True,
    ) -> None:
        train_data_loader = DataLoader(
            self._data["train"], batch_size=batch_size, shuffle=shuffle
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in tqdm(enumerate(train_data_loader)):
                inputs, labels = data["sentence"], data["nerc_tags"]

                optimizer.zero_grad()

                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(
                        f"[{epoch + 1}, {i + 1:5d}] avg_loss: {running_loss / 100:.3f}"
                    )
                    running_loss = 0.0

        print("Finished training")


class NERCTagger(Model):
    def __init__(
        self,
        dataset: datasets.DatasetDict,
        model_id: str = "michiyasunaga/BioLinkBERT-base",
    ) -> None:
        super().__init__(model_id)
        self.load_dataset(dataset)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.base_model.config.hidden_size, self.num_labels)

    def forward(self, input_data: dict) -> torch.Tensor:
        x = self.base_model(**input_data)
        x = F.relu(self.classifier(x.last_hidden_state))
        return x
