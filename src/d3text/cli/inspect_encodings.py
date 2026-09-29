"""Print what an encodings LMDB records, and the documents named."""

import argparse
import logging
import sys

import numpy

from d3text import encodings_store, logs


def main() -> None:
    """Print the store's stamps and document count, then each named key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="the encodings store's directory")
    parser.add_argument(
        "keys", nargs="*", help="document keys whose arrays to print"
    )
    args = parser.parse_args()
    # The report is the command's whole output, so a level quieting a sweep
    # must not silence it.
    logger = logs.configure(logging.INFO)

    with encodings_store.EncodingsStore(args.path) as store:
        try:
            provenance = encodings_store.read_provenance(store)
        except ValueError as refused:
            logger.info(f"{args.path}: {refused}")
        else:
            logger.info(
                f"{args.path}: no provenance recorded"
                if provenance is None
                else f"{args.path}: {provenance.base_model}, max_length "
                f"{provenance.max_length}, stride {provenance.stride}"
            )
        digest = encodings_store.read_content_digest(store)
        logger.info(f"content digest: {digest or 'not recorded'}")
        logger.info(f"{len(store.keys()):,} documents")

        for key in args.keys:
            encoding = store.get(key)
            if encoding is None:
                logger.info(f"{key}: not in the store")
                continue
            logger.info(f"{key}: {encoding['input_ids'].shape[0]} windows")
            with numpy.printoptions(threshold=sys.maxsize, linewidth=120):
                for name, array in encoding.items():
                    logger.info(f"  {name}:\n{array}")


if __name__ == "__main__":
    main()
