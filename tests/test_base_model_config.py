"""No module invents a field on the base model's `transformers` config.

A `PretrainedConfig` is a plain object: assigning an attribute it does not
define stores it and nothing ever reads it back, so the failure is silent by
construction and this is a source-level check. The scan follows a local alias,
`setattr`, `.update({...})` and tuple targets, but deliberately not a bare
`model.config.foo`: in this package `self.config` is `ModelConfig`, whose
fields are ours to add to, and matching on attribute name alone would trade a
hypothetical evasion for a real false-positive source.
"""

import ast
import pathlib

import transformers

from d3text.models.config import ModelConfig

REPO_ROOT = pathlib.Path(__file__).parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "d3text"
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _is_base_model_config_expr(expr: ast.expr) -> bool:
    """True for the literal chain `<x>.base_model.config`."""
    if not (isinstance(expr, ast.Attribute) and expr.attr == "config"):
        return False
    owner = expr.value
    if isinstance(owner, ast.Attribute):
        return owner.attr == "base_model"
    if isinstance(owner, ast.Name):
        return owner.id == "base_model"
    return False


def _flatten_targets(targets: list[ast.expr]) -> list[ast.expr]:
    """Tuple/list assignment targets unpacked to their individual elements."""
    flat: list[ast.expr] = []
    for target in targets:
        if isinstance(target, (ast.Tuple, ast.List)):
            flat.extend(_flatten_targets(list(target.elts)))
        else:
            flat.append(target)
    return flat


def _assigned_config_fields(tree: ast.AST) -> list[tuple[int, str]]:
    """Names assigned onto the base model's config, with line numbers.

    Only the `base_model.config` spelling, or a local alias bound from it
    within the same function, is matched.
    """
    found: list[tuple[int, str]] = []

    def is_config_ref(expr: ast.expr, aliases: set[str]) -> bool:
        if _is_base_model_config_expr(expr):
            return True
        return isinstance(expr, ast.Name) and expr.id in aliases

    def check_write_target(target: ast.expr, aliases: set[str]) -> None:
        if not isinstance(target, ast.Attribute):
            return
        if is_config_ref(target.value, aliases):
            found.append((target.lineno, target.attr))

    def check_setattr_call(call: ast.Call, aliases: set[str]) -> None:
        func = call.func
        if not (isinstance(func, ast.Name) and func.id == "setattr"):
            return
        if len(call.args) < 2:
            return
        name_arg = call.args[1]
        if not (
            is_config_ref(call.args[0], aliases)
            and isinstance(name_arg, ast.Constant)
            and isinstance(name_arg.value, str)
        ):
            return
        found.append((call.lineno, name_arg.value))

    def check_update_call(call: ast.Call, aliases: set[str]) -> None:
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "update"):
            return
        if not is_config_ref(func.value, aliases):
            return
        for arg in call.args:
            if not isinstance(arg, ast.Dict):
                continue
            for key in arg.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    found.append((call.lineno, key.value))
        for kw in call.keywords:
            if kw.arg is not None:
                found.append((call.lineno, kw.arg))

    def scan(node: ast.AST, aliases: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # A local alias is only ever local to the function that
                # binds it, so nested functions get a fresh alias set.
                scan(child, set())
                continue

            if isinstance(child, ast.Assign):
                for target in _flatten_targets(child.targets):
                    check_write_target(target, aliases)
                if (
                    len(child.targets) == 1
                    and isinstance(child.targets[0], ast.Name)
                    and is_config_ref(child.value, aliases)
                ):
                    aliases.add(child.targets[0].id)
            elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
                check_write_target(child.target, aliases)
            elif isinstance(child, ast.Call):
                check_setattr_call(child, aliases)
                check_update_call(child, aliases)

            scan(child, aliases)

    scan(tree, set())
    return found


def _reference_config() -> transformers.PretrainedConfig:
    """The `transformers` config `load_base_model` would actually build.

    Derived from the configured base model rather than a hardcoded class, since
    `load_base_model` resolves through `AutoConfig`. Falls back to `BertConfig`
    when the base model is not reachable, so the scan stays runnable offline.
    """
    base_model = ModelConfig().base_model
    try:
        return transformers.AutoConfig.from_pretrained(base_model)
    except Exception:
        return transformers.BertConfig()


def test_no_invented_base_model_config_fields() -> None:
    reference = _reference_config()
    offenders = []
    files_scanned = 0
    for root in (PACKAGE_ROOT, SCRIPTS_ROOT):
        for path in sorted(root.rglob("*.py")):
            files_scanned += 1
            tree = ast.parse(path.read_text())
            for lineno, field in _assigned_config_fields(tree):
                if not hasattr(reference, field):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: {field}"
                    )
    assert files_scanned, "scanned zero files — check PACKAGE_ROOT/SCRIPTS_ROOT"
    assert not offenders, (
        "assigned to config fields `transformers` does not define, so nothing "
        f"reads them back: {offenders}"
    )


def _fields_in(source: str) -> list[tuple[int, str]]:
    return _assigned_config_fields(ast.parse(source))


def test_scan_catches_a_write_through_a_local_alias() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    cfg = self.base_model.config
    cfg.use_memory_efficient_attention = True
"""
    )
    assert fields == [(4, "use_memory_efficient_attention")]


def test_scan_catches_setattr_on_the_config() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    setattr(self.base_model.config, "use_memory_efficient_attention", True)
"""
    )
    assert fields == [(3, "use_memory_efficient_attention")]


def test_scan_catches_setattr_through_a_local_alias() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    cfg = self.base_model.config
    setattr(cfg, "use_memory_efficient_attention", True)
"""
    )
    assert fields == [(4, "use_memory_efficient_attention")]


def test_scan_catches_a_tuple_target() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    self.base_model.config.use_memory_efficient_attention, x = True, 1
"""
    )
    assert fields == [(3, "use_memory_efficient_attention")]


def test_scan_catches_an_update_call_on_the_config() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    self.base_model.config.update({"use_memory_efficient_attention": True})
"""
    )
    assert fields == [(3, "use_memory_efficient_attention")]


def test_scan_catches_an_update_call_through_a_local_alias() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    cfg = self.base_model.config
    cfg.update({"use_memory_efficient_attention": True})
"""
    )
    assert fields == [(4, "use_memory_efficient_attention")]


def test_scan_catches_an_update_call_with_keyword_arguments() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    self.base_model.config.update(use_memory_efficient_attention=True)
"""
    )
    assert fields == [(3, "use_memory_efficient_attention")]


def test_scan_does_not_flag_a_bare_model_config_write() -> None:
    # No `base_model` anywhere in the chain: `model.config` here is a
    # different object than the transformers `PretrainedConfig` reached
    # through `base_model.config`, and this codebase never writes the
    # transformers config through that shorter spelling (see module
    # docstring) — extending the matcher to it would instead start flagging
    # legitimate writes to a plain `ModelConfig`.
    fields = _fields_in(
        """
def configure(model):
    model.config.foo = True
"""
    )
    assert fields == []


def test_scan_does_not_flag_ordinary_reads() -> None:
    fields = _fields_in(
        """
def load_base_model(self):
    cfg = self.base_model.config
    hidden = cfg.hidden_size
    if self.base_model.config.hidden_size > 0:
        pass
    return hidden
"""
    )
    assert fields == []


def test_scan_does_not_flag_unrelated_config_variables() -> None:
    # `self.config` is `ModelConfig` in this package, not the base model's
    # transformers config, and a plain local named `config` elsewhere is not
    # an alias unless it was actually bound from `….base_model.config`.
    fields = _fields_in(
        """
def configure(self):
    config = self.config
    config.entity_threshold = 0.9
"""
    )
    assert fields == []


def test_scan_keeps_aliases_scoped_to_their_own_function() -> None:
    # `cfg` aliases the base model's config in the first function only; the
    # same name in a second function is an unrelated local.
    fields = _fields_in(
        """
def load_base_model(self):
    cfg = self.base_model.config

def unrelated(self):
    cfg = self.config
    cfg.entity_threshold = 0.9
"""
    )
    assert fields == []
