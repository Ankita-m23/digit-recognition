import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from digit_recognition import load_model, normalize, DEVICE


def predict_multi(path):
    model = load_model()
    arr = np.array(Image.open(path).convert("L"))
    if arr.mean() > 127:  # light background -> white digits on black
        arr = 255 - arr
    H, W = arr.shape

    # find separate blobs of ink
    mask = ndimage.binary_dilation(arr > 100, iterations=2)
    labels, _ = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if (y1 - y0) < 0.03 * H and (x1 - x0) < 0.03 * W:
            continue
        boxes.append([x0, y0, x1, y1])

    # drop frames that surround another box
    def contains(a, b):
        return a is not b and a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]
    boxes = [b for b in boxes if not any(contains(b, o) for o in boxes)]
    if not boxes:
        print("No digits found in the image.")
        return

    # reading order: rows top to bottom, left to right inside a row
    med_h = float(np.median([b[3] - b[1] for b in boxes]))
    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
    rows, current = [], [boxes[0]]
    for b in boxes[1:]:
        if abs((b[1] + b[3]) / 2 - (current[-1][1] + current[-1][3]) / 2) < 0.6 * med_h:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    rows = [sorted(r, key=lambda b: b[0]) for r in rows]
    ordered = [b for r in rows for b in r]

    # make 28x28 MNIST-style inputs and predict
    tensors = []
    for x0, y0, x1, y1 in ordered:
        crop = Image.fromarray(arr[y0:y1, x0:x1])
        w, h = crop.size
        scale = 20.0 / max(w, h)
        crop = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
        tensors.append(normalize(canvas))
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)
    preds = probs.argmax(1).tolist()
    confs = probs.max(1).values.tolist()

    idx = 0
    for r in rows:
        print(" ".join(str(preds[idx + i]) for i in range(len(r))))
        idx += len(r)
    print("\nDetails:")
    for i, (d, c) in enumerate(zip(preds, confs), 1):
        flag = "" if c >= 0.9 else "   <- low confidence"
        print(f"  #{i}: {d}  ({c * 100:.1f}%){flag}")

    # annotated picture
    fig, ax = plt.subplots(figsize=(6, 6 * H / W + 0.5))
    ax.imshow(arr, cmap="gray")
    for (x0, y0, x1, y1), d, c in zip(ordered, preds, confs):
        color = "lime" if c >= 0.9 else "orange"
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=color, linewidth=2))
        ax.text(x0, max(y0 - 3, 0), str(d), color=color,
                fontsize=14, fontweight="bold", va="bottom")
    ax.axis("off")
    plt.tight_layout()
    out_name = f"multi_{time.strftime('%H%M%S')}.png"
    plt.savefig(os.path.abspath(out_name))
    print(f"\nSaved {out_name}")
    if hasattr(os, "startfile"):
        os.startfile(os.path.abspath(out_name))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multi_digits.py your_image.png")
    else:
        predict_multi(sys.argv[1])