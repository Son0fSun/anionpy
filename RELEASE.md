# Release notes for maintainers

## This tree is the public ship set

- No private monorepo `.git` history is included.
- Research crates that are not on the `ionp-py` dependency path are omitted.
- `tools/check_public_export.py` must exit 0 before any tag or push.

## First-time publish (already applied when this tree was created)

```bash
cd /path/to/anionpy-public
python3 tools/check_public_export.py   # must pass
git init
git add -A
git commit -m "Initial public release of anionpy 0.1.0"
# add remote and push only after human review
```

## Verify before push

```bash
python3 tools/check_public_export.py
cargo test --workspace --release
# optional: maturin develop --release && differential suite
```

## Version

`pyproject.toml` / workspace version: **0.1.0** (alpha).
