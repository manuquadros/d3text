import sys

import pytest
from d3text.cli import calibrate_relations


def test_the_input_checkpoint_is_never_overwritten(tmp_path, monkeypatch):
    """The checkpoint is the expensive artefact; a calibrated copy written
    over it would leave nothing to recalibrate from if the run were wrong."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"")
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate-relations", "config.toml", str(model), str(model)],
    )

    with pytest.raises(SystemExit, match="output is the input checkpoint"):
        calibrate_relations.main()
