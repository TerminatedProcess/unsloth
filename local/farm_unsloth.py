#!/usr/bin/env python3
"""Build an Unsloth Studio "farm" of links pointing into the HubRoot model store.

One physical copy stays in /mnt/llm/hub/hubmodels and Studio reaches it through
links. Nothing is ever copied.

Weights are HARD links, not symlinks: resolve_local_gguf_child() resolves symlinks
before its containment check, so a symlinked weight fails to load with
"gguf_filename must resolve to a file inside the repo". A hard link IS the file and
costs no disk, but requires the hub and the farm to share one filesystem, and means
deleting the hub copy does NOT free space until the farm entry goes too -- which is
what --rebuild is for. LoRAs stay symlinks; their loader has no containment check.

Layout is dictated by Unsloth Studio's scanner, which is fussier than ComfyUI's:

  * A custom scan folder is walked ONE level deep. A top-level child registers only
    if it is a directory holding an immediate weight file, or is itself a ``.gguf``.
    A loose top-level ``.safetensors`` is skipped -- hence one dir per checkpoint.
  * A scan folder yields at most 200 models (_MAX_MODELS_PER_CUSTOM_FOLDER) and
    stops walking after 2000 entries, both silently. 204 all-in-one checkpoints
    therefore have to be split across several folders, one per base model.
  * Diffusion LoRAs use a DIFFERENT scanner (core/inference/diffusion_lora.py):
    one flat directory, loose files, no per-folder cap. A ``<stem>.json`` sidecar
    declares the family so the picker filters LoRAs to the loaded model.

Lives in this fork (on the `mryan` branch) under local/, which upstream never
creates, so a `gitupdate` rebase can never conflict with it. The `rebuildfarm`
alias in .salias_f runs it.

Usage:
    python3 local/farm_unsloth.py                 # plan only (default; writes nothing)
    python3 local/farm_unsloth.py --apply         # after ADDING models to the hub
    python3 local/farm_unsloth.py --apply --rebuild   # after DELETING; frees pinned space
    python3 local/farm_unsloth.py --apply --only checkpoints,gguf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HUBROOT_API = os.environ.get("HUBROOT_API", "http://localhost:8000")
HUB_MODELS = Path(os.environ.get("HUBROOT_MODELS", "/mnt/llm/hub/hubmodels/models"))
FARM_ROOT = Path(os.environ.get("UNSLOTH_FARM", "/mnt/llm/unsloth"))
STUDIO_LORAS = Path(
    os.environ.get("UNSLOTH_LORAS", Path.home() / ".unsloth/studio/loras/diffusion")
)

# Studio caps a scan folder at 200 models. Stay clearly under it so a few new
# imports don't silently push a folder over the edge.
FOLDER_SOFT_CAP = 180

# HubRoot base_model -> the folder its checkpoints are farmed into. Anything
# unlisted lands in "misc", which keeps rare bases out of the big folders.
CHECKPOINT_BUCKET = {
    "illustrious": "illustrious",
    "sdxl 1.0": "sdxl",
    "pony": "pony",
    "noobai": "illustrious",  # a NoobAI checkpoint is an Illustrious derivative
}

# HubRoot base_model -> Unsloth Studio diffusion family name (diffusion_families.py).
# Written into each LoRA's sidecar so the picker family-gates it. A base with no
# mapping is skipped rather than deployed unfamilied: an unfamilied entry shows up
# for EVERY model, which is what makes a big LoRA list unusable.
LORA_FAMILY = {
    "illustrious": "sdxl",
    "pony": "sdxl",
    "noobai": "sdxl",
    "sdxl 1.0": "sdxl",
    "krea 2": "krea-2",
    "flux.1 d": "flux.1",
    "qwen": "qwen-image",
    "qwen-image": "qwen-image",
    "zimageturbo": "z-image",
}

WEIGHT_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth")

# HubRoot base_model -> the Unsloth diffusion family token appended to each
# checkpoint's directory name. Studio identifies a bare single-file checkpoint by
# NAME (detect_family), matching a family name/alias as a DELIMITED token -- so
# "analogMadnessSDXL_xl5" does not match sdxl even though it contains the letters,
# and every one of the 204 checkpoints resolved to task=null and was hidden from
# the Images picker. A trailing "-sdxl" fixes it.
CHECKPOINT_FAMILY = {
    "illustrious": "sdxl",
    "pony": "sdxl",
    "noobai": "sdxl",
    "sdxl 1.0": "sdxl",
    "krea 2": "krea-2",
    "flux.1 d": "flux.1",
    "qwen-image": "qwen-image",
    "chroma": None,  # no Unsloth family; it will list but stay task=null
}

# Studio's inventory classifies a directory holding a loose .safetensors and no
# config.json as model_format="unknown", and _scan_custom_folder drops every
# format outside {gguf, safetensors, adapter}. An empty config.json is the whole
# difference between "safetensors" (kept) and "unknown" (dropped). It must NOT be
# model_index.json: resolve_local_single_file returns None for any directory
# holding one, which would stop the single-file reinterpretation and 400 the load.
CHECKPOINT_CONFIG_JSON = "{}\n"


def fetch_models() -> list[dict]:
    out: list[dict] = []
    offset = 0
    while True:
        url = f"{HUBROOT_API}/api/models?limit=1000&offset={offset}"
        with urllib.request.urlopen(url, timeout = 120) as fh:
            batch = json.load(fh).get("models") or []
        out += batch
        if len(batch) < 1000:
            return out
        offset += 1000


def slugify(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "").strip())
    # " - " in a civitai title would otherwise leave "name---model".
    slug = re.sub(r"-{2,}", "-", slug).strip("-._").lower()
    return slug or "model"


def source_of(model: dict) -> Path:
    return HUB_MODELS / model["hash_blake3"] / model["filename"]


def display_slug(model: dict, used: set[str]) -> str:
    """A unique directory name -- the FILENAME stem, not HubRoot's title.

    Studio takes the display name from the directory name verbatim (it never
    resolves the symlink), so this is what shows in the picker and what search
    matches. The CivitAI title is not usable for it: one title covers every
    version of a model, so "GonzaLomo XL/Flux/Pony" collapsed v20PonyDMD,
    v60PhotoXLDMD, v60PhotoXLNonDMD and v70PhotoXL into one name plus numeric
    suffixes, and searching the filename found nothing. The stem carries the
    version, is already filesystem-safe, and matches what ComfyUI and HubRoot
    show for the same file.
    """
    # Case-preserving: the stem is already a real filename, and lowercasing
    # "gonzalomoXLFluxPony_v70PhotoXL" only makes it harder to read.
    base = re.sub(r"[/\\]+", "-", Path(model["filename"]).stem).strip() or "model"
    slug, n = base, 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


class Plan:
    def __init__(self) -> None:
        # dest -> (target, sidecar_text_or_None, hardlink?)
        self.links: dict[Path, tuple[Path, str | None, bool]] = {}
        self.folders: list[Path] = []
        self.skipped: Counter = Counter()
        # Plain files written next to a link (a checkpoint dir's config.json).
        self.files: dict[Path, str] = {}

    def add(
        self, dest: Path, target: Path, sidecar: str | None = None, *, hard: bool = False
    ) -> None:
        self.links[dest] = (target, sidecar, hard)


def plan_checkpoints(models: list[dict], plan: Plan) -> None:
    """All-in-one checkpoints: one directory per model, bucketed by base.

    Only is_all_in_one == 1. Those carry UNet + text encoder + VAE in the single
    file, and Studio's SDXL family sets single_file_is_pipeline, so they load with
    nothing fetched. A bare DiT (is_all_in_one == 0) still needs its family's text
    encoder and VAE from the base repo, so farming it does not make it offline.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for m in models:
        if m.get("model_type") != "checkpoint" or m.get("is_all_in_one") != 1:
            continue
        buckets[CHECKPOINT_BUCKET.get(str(m.get("base_model")).strip().lower(), "misc")].append(m)

    for bucket, entries in sorted(buckets.items()):
        entries.sort(key = lambda m: (m.get("name") or m["filename"]).lower())
        # Split an oversized bucket rather than let Studio truncate it in silence.
        chunks = [
            entries[i : i + FOLDER_SOFT_CAP] for i in range(0, len(entries), FOLDER_SOFT_CAP)
        ]
        for idx, chunk in enumerate(chunks):
            name = bucket if len(chunks) == 1 else f"{bucket}-{idx + 1}"
            folder = FARM_ROOT / "checkpoints" / name
            plan.folders.append(folder)
            used: set[str] = set()
            for m in chunk:
                src = source_of(m)
                if not src.exists():
                    plan.skipped["missing_source"] += 1
                    continue
                # Case-insensitive: HubRoot stores "Qwen-Image" but "sdxl 1.0",
                # and an exact-match lookup silently drops the odd one out.
                family = CHECKPOINT_FAMILY.get(str(m.get("base_model")).strip().lower())
                slug = display_slug(m, used)
                if family:
                    slug = f"{slug}-{family}"
                else:
                    plan.skipped["no_family_token"] += 1
                model_dir = folder / slug
                plan.add(model_dir / m["filename"], src, hard = True)
                plan.files[model_dir / "config.json"] = CHECKPOINT_CONFIG_JSON


def plan_gguf(models: list[dict], plan: Plan) -> None:
    """Every GGUF, flat. A bare ``.gguf`` is the one loose top-level file the
    scan-folder walk accepts, so these need no per-model directory."""
    folder = FARM_ROOT / "gguf"
    plan.folders.append(folder)
    seen: set[str] = set()
    for m in sorted(models, key = lambda m: m["filename"].lower()):
        if not m["filename"].lower().endswith(".gguf"):
            continue
        src = source_of(m)
        if not src.exists():
            plan.skipped["missing_source"] += 1
            continue
        name = m["filename"]
        if name in seen:  # same filename, different hash
            name = f"{Path(name).stem}-{m['hash_blake3'][:8]}.gguf"
        seen.add(name)
        plan.add(folder / name, src, hard = True)


def plan_loras(models: list[dict], plan: Plan) -> None:
    """LoRAs go flat into Studio's own LoRA dir, each with a family sidecar."""
    for m in sorted(models, key = lambda m: m["filename"].lower()):
        if m.get("model_type") not in ("lora", "locon"):
            continue
        family = LORA_FAMILY.get(str(m.get("base_model")).strip().lower())
        if not family:
            plan.skipped["lora_unmapped_family"] += 1
            continue
        src = source_of(m)
        if not src.exists():
            plan.skipped["missing_source"] += 1
            continue
        dest = STUDIO_LORAS / m["filename"]
        sidecar = json.dumps({"family": family, "weight_default": 1.0}, indent = 2)
        plan.add(dest, src, sidecar)


def _same_file(a: Path, b: Path) -> bool:
    """True when *a* is already a hard link to *b* (same inode + device)."""
    try:
        sa, sb = a.stat(follow_symlinks = False), b.stat()
    except OSError:
        return False
    return sa.st_ino == sb.st_ino and sa.st_dev == sb.st_dev


def apply(plan: Plan) -> tuple[int, int, int]:
    created = updated = unchanged = 0
    for dest, (target, sidecar, hard) in sorted(plan.links.items()):
        dest.parent.mkdir(parents = True, exist_ok = True)
        if hard:
            # HARD link, not symbolic: resolve_local_gguf_child() resolves symlinks
            # BEFORE its containment check ("gguf_filename must resolve to a file
            # inside the repo"), so a symlinked weight pointing into the hub is
            # rejected at load with no way to opt out. A hard link IS the file, so
            # it passes -- and costs no disk, both paths sharing one inode. Requires
            # hub and farm on ONE filesystem; /mnt/llm holds both.
            if _same_file(dest, target):
                unchanged += 1
            else:
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                try:
                    os.link(target, dest)
                except OSError as exc:
                    print(
                        f"  ! hard link failed ({exc.strerror}): {dest}\n"
                        f"    hub and farm must share a filesystem; a symlink here "
                        f"would list but fail to load.",
                        file = sys.stderr,
                    )
                    continue
                created += 1
        elif dest.is_symlink():
            if Path(os.readlink(dest)) == target:
                unchanged += 1
            else:
                dest.unlink()
                dest.symlink_to(target)
                updated += 1
        elif dest.exists():
            # A real file here is not ours to replace.
            print(f"  ! refusing to replace real file: {dest}", file = sys.stderr)
            continue
        else:
            dest.symlink_to(target)
            created += 1
        if sidecar is not None:
            side = dest.with_suffix(".json")
            if not side.exists() or side.read_text(encoding = "utf-8") != sidecar:
                side.write_text(sidecar, encoding = "utf-8")
    for path, text in sorted(plan.files.items()):
        path.parent.mkdir(parents = True, exist_ok = True)
        if not path.exists() or path.read_text(encoding = "utf-8") != text:
            path.write_text(text, encoding = "utf-8")
    return created, updated, unchanged


def destroy(only: set[str]) -> None:
    """Tear the farm down so the next build reflects deletions from the hub.

    An incremental run only ever ADDS, so a model removed from the hub leaves its
    entry behind -- and because the weights are hard links, that stale entry also
    pins the blob's disk space even after HubRoot purges it. Rebuilding is the
    only way to actually reclaim it.

    Checkpoint and GGUF folders are farm-exclusive, so they go wholesale. The LoRA
    directory is NOT ours alone: Studio's trainer publishes adapters straight into
    it, so only symlinks (and their sidecars) are removed and real files survive.
    """
    for name in ("checkpoints", "gguf"):
        if name not in only:
            continue
        target = FARM_ROOT / name
        # Refuse to recurse anywhere unexpected, however FARM_ROOT was overridden.
        if target.is_dir() and target.resolve().is_relative_to(FARM_ROOT.resolve()):
            shutil.rmtree(target)
            print(f"  destroyed  {target}")

    if "loras" in only and STUDIO_LORAS.is_dir():
        removed = 0
        for entry in sorted(STUDIO_LORAS.iterdir()):
            if not entry.is_symlink():
                continue  # a trainer-published adapter, not ours
            sidecar = entry.with_suffix(".json")
            entry.unlink()
            if sidecar.is_file() and not sidecar.is_symlink():
                sidecar.unlink()
            removed += 1
        print(f"  destroyed  {removed} lora links in {STUDIO_LORAS} (real files kept)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action = "store_true", help = "write links (default: plan only)")
    ap.add_argument("--only", default = "checkpoints,gguf,loras")
    ap.add_argument(
        "--rebuild",
        action = "store_true",
        help = "delete the farm first, so models removed from the hub disappear "
               "(and their hard-linked disk space is actually freed)",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    models = fetch_models()
    print(f"hub: {len(models)} models\n")

    plan = Plan()
    if "checkpoints" in only:
        plan_checkpoints(models, plan)
    if "gguf" in only:
        plan_gguf(models, plan)
    if "loras" in only:
        plan_loras(models, plan)

    per_folder: Counter = Counter()
    for dest in plan.links:
        per_folder[dest.parent if dest.parent in plan.folders else dest.parent.parent] += 1
    for folder in plan.folders:
        n = per_folder.get(folder, 0)
        flag = "  <-- OVER STUDIO'S 200 CAP" if n > 200 else ""
        print(f"  scan folder  {str(folder):<48} {n:>4} models{flag}")
    loras = sum(1 for d in plan.links if d.parent == STUDIO_LORAS)
    if loras:
        print(f"  lora dir     {str(STUDIO_LORAS):<48} {loras:>4} adapters (+ sidecars)")
    if plan.skipped:
        print("\n  skipped:", dict(plan.skipped))
    print(f"\n  total links: {len(plan.links)}")

    if not args.apply:
        if args.rebuild:
            print("\n(--rebuild would delete the farm first)")
        print("\n(plan only -- re-run with --apply to write)")
        return 0
    if args.rebuild:
        print()
        destroy(only)
    created, updated, unchanged = apply(plan)
    print(f"\napplied: {created} created, {updated} retargeted, {unchanged} already correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
