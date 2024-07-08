import s800classed
import datasets


def species800(
        train_size: float = 0.15,
        dev_size: float = 0.15,
        seed: int = 4
) -> datasets.Dataset:
    return s800classed.load(train_size, dev_size, seed)
