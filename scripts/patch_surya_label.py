#!/usr/bin/env python3
"""
Patch the installed surya package so the layout model never emits the `Form`
label. Anything surya would have called `<form>` becomes `ListItem` instead,
which lets marker's normal list path handle the region (including the
prefix-free itemization processors in marker.processors.list_line_explode
and marker.processors.list_gap_cluster).

Idempotent. Run once after `pip install surya-ocr` (or after upgrading).
"""
import importlib.util
import sys
from pathlib import Path


def main():
    spec = importlib.util.find_spec("surya.layout.label")
    if spec is None or spec.origin is None:
        print("ERROR: could not find surya.layout.label — is surya installed?", file=sys.stderr)
        sys.exit(1)

    path = Path(spec.origin)
    src = path.read_text()

    target_old = '"<form>": "Form",'
    target_new = '"<form>": "ListItem",'

    if target_new in src:
        print(f"already patched: {path}")
        return
    if target_old not in src:
        print(f"ERROR: expected line not found in {path}", file=sys.stderr)
        print("Surya may have changed its label map. Patch manually.", file=sys.stderr)
        sys.exit(1)

    path.write_text(src.replace(target_old, target_new))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
