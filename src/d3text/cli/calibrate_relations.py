"""Calibrate a trained checkpoint's relation decision thresholds.

Runs the checkpoint over the validation split, fits one threshold per typed
relation to maximise `relation_micro_f1_typed`, and writes a copy of the
checkpoint carrying them. `train` does the same for every checkpoint it
writes; this command is for one written before it did.
"""

import argparse
import dataclasses
import logging
import pathlib

from d3text import checkpoint, data, factory, runtime, token_labels
from d3text.cli.evaluate import load_evaluation_dataset
from d3text.models.config import load_model_config, token_labels_path
from d3text.models.ete import ETEBrendaModel
from d3text.schema import BRENDA_SCHEMA

logger = logging.getLogger(__name__)


def command_line_args() -> argparse.Namespace:
    """Parse the `calibrate-relations` command line.

    :return: the parsed arguments.
    :raises SystemExit: through `argparse`, on a malformed command line.
    """
    parser = argparse.ArgumentParser(
        prog="calibrate-relations", description=__doc__
    )
    parser.add_argument("config", help="the checkpoint's training config")
    parser.add_argument("checkpoint", help="the checkpoint to calibrate")
    parser.add_argument("output", help="where to write the calibrated copy")
    return parser.parse_args()


def main() -> None:
    """Calibrate `checkpoint` on the validation split and write `output`.

    :raises SystemExit: if `output` is `checkpoint` itself, or the config
        builds a model with no relation head.
    """
    args = command_line_args()
    if (
        pathlib.Path(args.output).resolve()
        == pathlib.Path(args.checkpoint).resolve()
    ):
        raise SystemExit(
            "COULD NOT RUN: output is the input checkpoint; write the "
            "calibrated copy elsewhere rather than overwrite the original"
        )
    config = load_model_config(args.config)
    runtime.configure(seed=config.seed)

    saved = checkpoint.load(args.checkpoint)
    labels_path = token_labels_path(config)
    dataset = load_evaluation_dataset(
        config_base_model=config.base_model,
        vocabulary=saved.vocabulary,
        tokenizer=token_labels.store_tokenizer_stamp(labels_path),
        split="val",
    )
    model = factory.build_model_for_dataset(config, BRENDA_SCHEMA, dataset)
    if not isinstance(model, ETEBrendaModel):
        raise SystemExit(
            f"COULD NOT RUN: {config.model_class} has no relation head, so "
            "it has no relation decision to calibrate"
        )
    model.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    model.load_state_dict(saved.state_dict)
    model.to(model.device)

    calibration = model.calibrate_relation_thresholds(
        data.get_batch_loader(
            dataset=dataset.data["val"],
            batch_size=1,
            max_chunks=config.batch_max_chunks,
        )
    )
    for key, value in calibration.metrics().items():
        logger.info("%s: %.4f", key, value)
    if calibration.thresholds is None:
        logger.info(
            "No threshold setting beat the argmax on validation; the copy "
            "keeps deciding relations by argmax."
        )

    calibrated = dataclasses.replace(
        saved, relation_thresholds=calibration.thresholds
    )
    checkpoint.save(
        args.output,
        **{
            field.name: getattr(calibrated, field.name)
            for field in dataclasses.fields(calibrated)
        },
    )
    logger.info("Calibrated checkpoint written to %s.", args.output)


if __name__ == "__main__":
    main()
