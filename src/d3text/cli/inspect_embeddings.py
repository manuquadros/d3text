"""Print what an embeddings LMDB records and what each sub-database holds."""

import argparse
import logging

from d3text import embeddings_store, logs


def _gib(size: int) -> str:
    return f"{size / 1024**3:.2f} GiB"


def main() -> None:
    """Print the env's provenance and one line per sub-database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="the base model's embeddings env")
    path = parser.parse_args().path
    # The report is the command's whole output, so a level quieting a sweep
    # must not silence it.
    logger = logs.configure(logging.INFO)

    description = embeddings_store.describe(path)
    provenance = description.provenance
    if description.provenance_error is not None:
        logger.info(f"{path}: {description.provenance_error}")
    elif provenance is None:
        logger.info(f"{path}: no provenance recorded")
    else:
        logger.info(
            f"{path}: {provenance.base_model}, max_length "
            f"{provenance.max_length}, stride {provenance.stride}, forward "
            f"dtype {provenance.forward_dtype or 'not recorded'}"
        )
    if not description.databases:
        logger.info("no sub-databases")
    for info in description.databases:
        ratio = "n/a" if info.ratio is None else f"{info.ratio:.2f}x"
        unknown = (
            f", {info.unknown_frames:,} unreadable"
            if info.unknown_frames
            else ""
        )
        logger.info(
            f"  {info.name}: {info.documents:,} documents; frames "
            f"{info.compressed_frames:,} compressed, {info.raw_frames:,} "
            f"raw{unknown}; ratio {ratio}; "
            f"{_gib(info.decompressed_bytes)} decompressed, "
            f"{_gib(info.disk_bytes)} on disk"
        )


if __name__ == "__main__":
    main()
