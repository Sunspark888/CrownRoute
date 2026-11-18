"""Write a synthetic input inventory for the C1 construction example."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def write_png(path, data):
    ok, encoded = cv2.imencode(".png", data)
    if not ok:
        raise OSError(path)
    encoded.tofile(path)


def make_demo(directory):
    directory.mkdir(parents=True, exist_ok=False)
    yy, xx = np.indices((96, 96))
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :, 1] = 80 + ((xx + yy) % 2) * 150
    write_png(directory / "texture.png", image)
    patches = []
    for name, x, y, width, height in [
        ("refine", 20, 20, 56, 56), ("rebuild", 35, 35, 18, 18),
        ("gate", 62, 35, 28, 28), ("direct", 1, 1, 16, 16),
        ("empty", 2, 2, 5, 5),
    ]:
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[y:y+height, x:x+width] = 255
        write_png(directory / f"{name}.png", mask)
        patches.append({
            "id": name, "image": "texture.png", "split": "train", "source_id": "synthetic",
            "boxes": [[20, 20, 76, 76]], "candidate_masks": [f"{name}.png"]
        })
    manifest = directory / "patches.json"
    manifest.write_text(json.dumps({"patches": patches}, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    make_demo(parser.parse_args().directory)
