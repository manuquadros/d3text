"""No module invents a field on the base model's `transformers` config.

A `PretrainedConfig` is a plain object: assigning an attribute it does not
define stores it and nothing ever reads it back. So a line like
`model.base_model.config.use_memory_efficient_attention = True` raises nothing,
changes nothing, and reads like a switch — the failure mode is silent by
construction, which is why this is a source-level check and not a behavioural
one. Attention is selected by `config._attn_implementation`, which
`transformers` resolves at load time.
"""

import ast
import pathlib

import transformers

PACKAGE_ROOT = pathlib.Path(__file__).parent.parent / "src" / "d3text"


def _assigned_config_fields(
    tree: ast.AST,
) -> list[tuple[int, str]]:
    """Names assigned onto a `…base_model.config` attribute, with line numbers.

    Only the `base_model.config` spelling is matched: `self.config` in this
    package is `ModelConfig`, whose fields are ours to add to.
    """
    found = []
    for node in ast.walk(tree):
        targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            owner = target.value
            if not (
                isinstance(owner, ast.Attribute) and owner.attr == "config"
            ):
                continue
            model = owner.value
            model_name = (
                model.attr
                if isinstance(model, ast.Attribute)
                else model.id
                if isinstance(model, ast.Name)
                else None
            )
            if model_name == "base_model":
                found.append((target.lineno, target.attr))
    return found


def test_no_invented_base_model_config_fields() -> None:
    reference = transformers.BertConfig()
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for lineno, field in _assigned_config_fields(tree):
            if not hasattr(reference, field):
                offenders.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {field}"
                )
    assert not offenders, (
        "assigned to config fields `transformers` does not define, so nothing "
        f"reads them back: {offenders}"
    )
