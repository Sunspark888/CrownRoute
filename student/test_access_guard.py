from __future__ import annotations

import builtins
from pathlib import Path


FORBIDDEN_NAMES = {
    "test.txt",
    "gt_test_samples.json",
    "gt_test_manual_coco.json",
    "normalized_gt_test_manual_coco.json",
    "test_ap_table.csv",
    "fixed_threshold_test_metrics.json",
}


def _is_forbidden(value: object) -> bool:
    try:
        path = Path(value)
    except TypeError:
        return False
    name = path.name.lower()
    normalized = str(path).replace("\\", "/").lower()
    return (
        name in FORBIDDEN_NAMES
        or "/split/test.txt" in normalized
        or "/predictions/test/" in normalized
        or "/manual_polygon_json/test" in normalized
    )


def install_test_access_guard() -> None:
    original_builtin_open = builtins.open
    original_path_open = Path.open

    def guarded_builtin_open(file, *args, **kwargs):
        if _is_forbidden(file):
            raise RuntimeError(f"TEST_FORBIDDEN_FILE_ACCESS={file}")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_path_open(self: Path, *args, **kwargs):
        if _is_forbidden(self):
            raise RuntimeError(f"TEST_FORBIDDEN_FILE_ACCESS={self}")
        return original_path_open(self, *args, **kwargs)

    builtins.open = guarded_builtin_open
    Path.open = guarded_path_open
