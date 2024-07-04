import os

import datasets
import sklearn
import torch
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Model(torch.nn.Module):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self._model = transformers.AutoModel.from_pretrained(model_id)

    def load(self, dataset: datasets.DatasetDict) -> None:
        """
        Load dataset and tokenize it, keeping track of NERC tags.
        """

        max_length = max(len(sample["tokens"]) for sample in dataset["train"])

        data = dataset.map(
            lambda sample: {"sentence": self.tokenize_and_align(sample, max_length)},
            remove_columns=["tokens", "nerc_tags"],
        )

        # One-hot encoding of the NERC labels
        self._label_encoder = sklearn.preprocessing.LabelBinarizer()
        self._label_encoder.fit(
            [
                label
                for sample in data["train"]
                for label in sample["sentence"]["nerc_tags"]
            ]
        )
        data = data.map(
            lambda sample: {
                "nerc_tags": self._label_encoder.transform(
                    sample["sentence"]["nerc_tags"]
                )
            }
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._data = data.with_format("torch", device=device)

    def tokenize_and_align(
        self, sample: dict[str, list[str]], max_length: int
    ) -> dict[str, list[str]]:
        tokenized = self._tokenizer(
            sample["tokens"],
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length,
        )

        labels = []
        for idx in tokenized.word_ids():
            if idx is None:
                labels.append("#")
            else:
                labels.append(sample["nerc_tags"][idx])

        tokenized["nerc_tags"] = labels

        return tokenized


class BioLinkBert(Model):
    def __init__(self) -> None:
        super().__init__("michiyasunaga/BioLinkBERT-large")


class NERCTagger(Model):
    def __init__(self):
        super().__init__()

    def forward(input):
        y = self.flatten
