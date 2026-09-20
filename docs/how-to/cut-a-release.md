# Cut a release

A release is a tag. `pdm-backend` reads the tag at build time and
`pyproject.toml` carries no version number, so creating the tag is the whole
act — there is no file to edit and no number to keep in sync.

Tag when you are about to *cite* the code: a submission, a shared checkpoint,
a set of results someone else will quote. There is no cadence to keep, because
nothing breaks between tags — every run records `git_describe`, so a run made
forty commits after `v0.2.0` identifies itself as `v0.2.0-40-gabc1234` and
stays exactly reproducible.

## Look before you cut

```bash
pdm run next-version      # the version the commit subjects imply
pdm run release --dry-run # that version, plus the changelog section it would add
```

`--dry-run` changes nothing. It prints the section so you can check that the
commits since the last tag say what you think they say.

## Cut it

```bash
pdm run release
```

That renders `CHANGELOG.md`, commits it as `docs: changelog for X.Y.Z`, makes
an annotated tag whose message is the new section, and pushes the branch and
the tag together with `--atomic`. To stop before publishing:

```bash
pdm run release --no-push   # commit and tag locally; prints the push command
```

## When it declines

`release` exits **1** without changing anything when there is nothing to
release — no commits since the last tag, or every commit since it prefixed
`build:`, `chore:`, `ci:`, `test:` or `docs:`. Those cannot be what a reader
cites. A `!` break releases even under those prefixes, because dropping a
Python version is not invisible.

To release anyway, name the version yourself:

```bash
pdm run release --version v0.3.0
```

It exits **2** and changes nothing when it cannot proceed: HEAD is not on
`main`, the tag already exists, `CHANGELOG.md` has uncommitted edits, or
git-cliff is not installed. The one partial state it can leave — the changelog
committed but the tag not created — is reported in full, with the command to
finish by hand.

## What decides the version

`cliff.toml` maps commit prefixes onto changelog sections and onto the bump:
`feat:` takes the minor, `fix:`/`perf:`/`refactor:` the patch. A `!` break
takes the minor too rather than the major, because the project is still `0.x`
and has claimed no stability.

This is the one place the commit convention is load-bearing rather than
merely tidy — a `fix:` written as `chore:` silently becomes a release that
does not happen.
