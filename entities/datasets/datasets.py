import s800classed
import datasets


def species800(
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 4
) -> datasets.Dataset:
    return s800classed.load(test_size=test_size, val_size=val_size, seed=seed)
