#!/usr/bin/env python3
"""Delete QA screenshots so the next `@vl` pass can never read stale images.

The debug-loop / qa-loop flow captures screenshots, hands them to the `vl` vision
subagent, then MUST clear them: a later round that re-reads an old screenshot
produces findings for a build that no longer exists (this happened — see
data/qa_screenshots/round{1,2,3}/ which mixed 2-3 build generations).

Rule: after `vl` has read the images and findings are recorded, run this script.
It deletes IMAGE files only — markdown/JSON reports (AUDIT.md, manifest.json,
traversal-report.json, ...) are preserved.

Usage:
    python scripts/clean_screenshots.py            # dry-run (list what would go)
    python scripts/clean_screenshots.py --yes      # actually delete
    python scripts/clean_screenshots.py --yes --path data/qa_screenshots
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that accumulate QA screenshots (relative to repo root).
DEFAULT_TARGETS = [
    "data/qa_screenshots",
    "tests/visual_screenshots",
    ".playwright-mcp",
    "data/qa_loop/audit",
]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def iter_images(root_abs):
    for dirpath, _dirs, files in os.walk(root_abs):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--path", action="append", default=None,
                    help="limit to this relative path (repeatable); default: all targets")
    args = ap.parse_args()

    targets = args.path or DEFAULT_TARGETS
    found = []
    for rel in targets:
        root_abs = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isdir(root_abs):
            continue
        found.extend(iter_images(root_abs))

    if not found:
        print("[clean_screenshots] nothing to delete (no images found).")
        return 0

    total_bytes = 0
    for path in found:
        try:
            total_bytes += os.path.getsize(path)
        except OSError:
            pass
    mb = total_bytes / (1024 * 1024)

    verb = "DELETED" if args.yes else "WOULD DELETE"
    for path in found:
        print(f"  [{verb}] {os.path.relpath(path, PROJECT_ROOT)}")
    print(f"\n[clean_screenshots] {len(found)} image(s), {mb:.1f} MB — "
          + ("deleted." if args.yes else "dry-run (pass --yes to delete)."))

    if args.yes:
        for path in found:
            try:
                os.remove(path)
            except OSError as e:
                print(f"  [WARN] could not remove {path}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
