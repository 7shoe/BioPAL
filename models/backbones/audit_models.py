#!/usr/bin/env python3
"""Audit the local foundation-model caches against their Hugging Face counterparts.

Note that model support depends on the conda env (see ~/envs)

Checks per model, in every cache paths.yaml expects it in:
    presence   the models--<org>--<name> cache entry and a snapshot exist
    
    weights    resolved weight bytes (*.safetensors / pytorch_model*.bin;
               symlinks followed, broken links are failures); models marked
               expect_weights: false report the gap without failing
    
    revision   a pinned `revision` (if any) must be present as a snapshot
    
    freshness  the locally tracked commit (refs/main) vs the current upstream
               commit sha on Hugging Face (skipped with --offline)

Exit codes: 0 all OK, 1 any FAIL, 2 only WARNs.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import urllib.request
from datetime import date
from pathlib import Path

import yaml


def hf_upstream(repo_id: str) -> dict:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo_id}",
                                 headers={"User-Agent": "biopal-model-audit"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def weight_bytes(mdir: Path) -> tuple[int, int]:
    """(total resolved bytes of unique weight blobs, number of broken symlinks)"""
    blobs, broken = set(), 0
    for p in mdir.glob("snapshots/*/**/*"):
        if p.name.endswith(".safetensors") or (p.name.startswith("pytorch_model")
                                               and p.name.endswith(".bin")):
            if p.exists():
                blobs.add(p.resolve())
            else:
                broken += 1          # dangling symlink: blob was never fetched
    return sum(b.stat().st_size for b in blobs), broken


def fmt_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n / 1:.1f}{unit}"
        n /= 1024
    return f"{n}G"


def audit_model(name: str, spec: dict, base: Path, caches: dict,
                upstream: dict | None, meta: dict) -> list[str]:
    out = []
    expect_weights = spec.get("expect_weights", True)
    entry = "models--" + spec["repo_id"].replace("/", "--")

    for cache_key in spec["caches"]:
        mdir = base / caches[cache_key] / entry
        tag = f"{cache_key}:"
        if not mdir.is_dir():
            out.append(f"FAIL {tag} cache entry missing ({mdir})")
            continue
        snapshots = sorted(p.name for p in (mdir / "snapshots").glob("*") if p.is_dir())
        if not snapshots:
            out.append(f"FAIL {tag} no snapshots")
            continue

        nbytes, broken = weight_bytes(mdir)
        if broken:
            out.append(f"FAIL {tag} {broken} dangling weight symlinks (blobs not fetched)")
        if nbytes == 0 and expect_weights:
            out.append(f"FAIL {tag} no weight files")
        elif nbytes == 0:
            out.append(f"WARN {tag} no weights (known gap: expect_weights false "
                       "- download before using this model)")

        pin = spec.get("revision")
        if pin and pin not in snapshots:
            out.append(f"FAIL {tag} pinned revision {pin[:12]} not in snapshots "
                       f"{[s[:12] for s in snapshots]}")

        ref = mdir / "refs" / "main"
        local_sha = ref.read_text().strip() if ref.exists() else None
        meta.setdefault("revision", spec.get("revision") or local_sha)
        line = (f"    {tag:9s} weights={fmt_bytes(nbytes) if nbytes else '-'} "
                f"snapshots={[s[:12] for s in snapshots]}")
        if upstream is not None and (local_sha or pin):
            if upstream["sha"] in (local_sha, pin):
                line += "  (= upstream)"
            else:
                out.append(f"WARN {tag} upstream moved: local main {local_sha[:12]} "
                           f"!= upstream {upstream['sha'][:12]} "
                           f"(upstream modified {upstream.get('lastModified', '?')[:10]})")
        print(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path,
                    default=Path(__file__).resolve().parents[2] / "paths" / "paths.yaml")
    ap.add_argument("--machine", default=platform.node().split(".")[0])
    ap.add_argument("--models", nargs="*", help="subset of model names")
    ap.add_argument("--offline", action="store_true", help="skip upstream checks")
    ap.add_argument("--yaml", type=Path, metavar="FILE",
                    default=Path(__file__).parent / "models.yaml",
                    help="model-registry YAML to write (models whose status is OK; "
                         "default: models.yaml next to this script)")
    ap.add_argument("--no-yaml", action="store_true", help="audit only, write no registry")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.paths.read_text())
    base = (cfg["machines"].get(args.machine) or {}).get("models_base")
    if not base:
        sys.exit(f"No models_base for machine {args.machine!r}; known: {sorted(cfg['machines'])}")
    base = Path(base)
    print(f"machine={args.machine} base={base}\n")

    status, meta = {}, {}
    for name, spec in cfg["models"].items():
        if args.models and name not in args.models:
            continue
        print(f"== {name} ({spec['repo_id']}) [{spec['modality']}]")
        upstream = None
        if not args.offline:
            try:
                upstream = hf_upstream(spec["repo_id"])
            except Exception as e:
                print(f"    WARN upstream lookup failed: {type(e).__name__}: {e}")
        meta[name] = {"repo_id": spec["repo_id"], "modality": spec["modality"],
                      "caches": list(spec["caches"])}
        findings = audit_model(name, spec, base, cfg["caches"], upstream, meta[name])
        for f in findings:
            print(f"    {f}")
        status[name] = ("FAIL" if any(f.startswith("FAIL") for f in findings)
                        else "WARN" if findings else "OK")
        print(f"    -> {status[name]}\n")

    print(f"SUMMARY  {len(status)} models: "
          f"{sum(s == 'OK' for s in status.values())} OK, "
          f"{sum(s == 'WARN' for s in status.values())} WARN, "
          f"{sum(s == 'FAIL' for s in status.values())} FAIL")
    for name, s in status.items():
        if s != "OK":
            print(f"  {name}: {s}")

    if args.models and not args.no_yaml:
        print("\n[info] partial audit (--models): not writing the model registry")
    elif not args.no_yaml:
        ok = {name: meta[name] for name, s in status.items() if s == "OK"}
        header = ("# Model registry: foundation models that passed audit_models.py\n"
                  f"# Generated on {date.today()} "
                  f"(machine {args.machine}); do not edit -\n"
                  f"# regenerate with: python audit_models.py --yaml {args.yaml.name}\n"
                  "# `revision` is the pinned or locally verified commit; caches are\n"
                  "# keys into paths/paths.yaml, relative to the machine's models_base.\n")
        args.yaml.write_text(header + yaml.safe_dump({"models": ok}, sort_keys=True))
        print(f"\n[ok] wrote {args.yaml} ({len(ok)} models)")
    return 1 if "FAIL" in status.values() else 2 if "WARN" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
