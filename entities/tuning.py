import data
import models
import random
import torch._dynamo
torch.set_float32_matmul_precision("high")


ds = data.load_dataset(data.only_species_and_strains800())

for _ in range(3):
    configs = list(models.model_configs())
    random.shuffle(configs)
    
    for config in configs:
        nt = models.NERCTagger(ds, config=config)
        nt.cuda()

        torch._dynamo.reset()
        model = torch.compile(nt, mode="max-autotune", fullgraph=True)
        model.train_model()

        _, report = model.evaluate_model()
        print(report)
