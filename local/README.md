# local/ — personal tooling (not upstream)

Everything here belongs to the `mryan` branch of this fork and is **not** part of
upstream Unsloth. `local/` exists precisely because upstream has no directory by
that name, so `gitupdate`'s rebase of `mryan` onto `main` can never conflict with
it. Do not move these into `scripts/` — that dir is upstream's.

The repo's root `CLAUDE.md` is upstream's file. Personal notes go here or in the
`unsloth-memory` skill, never there, for the same rebase reason.

## farm_unsloth.py

Links models from the HubRoot store (`/mnt/llm/hub/hubmodels`) into a "farm" at
`/mnt/llm/unsloth` that Unsloth Studio can index — one physical copy of every
model, no duplication.

```bash
farmplan       # dry run; writes nothing
farmsync       # additive — after importing models into the hub
rebuildfarm    # destroy + rebuild — after deleting models
```

Aliases are in `.salias_f` at the repo root and resolve via `$UNSLOTH_REPO`, so
they work from any directory. Long form:
`python3 local/farm_unsloth.py [--apply] [--rebuild] [--only checkpoints,gguf,loras]`.

Three non-obvious rules the script encodes, each of which fails **silently** if
broken — the script's module docstring explains each in full:

1. **Weights are hard links, not symlinks.** `resolve_local_gguf_child()` resolves
   symlinks before its containment check, so a symlinked weight fails at load with
   *"gguf_filename must resolve to a file inside the repo"*. Requires hub and farm
   on one filesystem, and means a hub purge frees no space until the farm entry
   goes too — that is what `--rebuild` is for.
2. **Each checkpoint dir needs a `config.json`.** Without one the inventory calls
   it `model_format="unknown"` and drops it. Must not be `model_index.json`.
3. **The dir name must end in a family token** (`-sdxl`, `-krea-2`, …). Family
   detection matches a delimited token in the *name*, so `analogMadnessSDXL_xl5`
   does not match `sdxl` and would be hidden from the Images picker.

Full background — scanner limits, HF cache layout, load-performance behaviour,
rebuild-from-scratch — is in the `unsloth-memory` skill
(`~/.config/fish/claude-skills/unsloth-memory/`).
