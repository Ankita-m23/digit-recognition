"""
Handwritten Digit Recognition (MNIST) with a CNN in PyTorch.
 
Install:
    pip install torch torchvision pillow matplotlib
 
Usage:
    python digit_recognition.py train                 # train + evaluate, saves digit_cnn.pt
    python digit_recognition.py predict my_digit.png  # predict a digit from your own image
    python digit_recognition.py multi my_digits.png   # find and predict several digits in one image (needs scipy)
    python digit_recognition.py samples               # show predictions on random test images
"""
 
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps
 
MODEL_PATH = "digit_cnn.pt"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
 
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])
 
 
# ---------------------------------------------------------------- model
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.drop2(F.relu(self.fc1(x)))
        return self.fc2(x)
 
 
# ---------------------------------------------------------------- training
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total
 
 
def train(epochs=5, batch_size=128, lr=1e-3):
    # light augmentation helps with real handwriting
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_set = datasets.MNIST("data", train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=normalize)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)
 
    model = DigitCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
 
    print(f"Training on {DEVICE}")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % 100 == 0:
                print(f"  epoch {epoch} batch {i}/{len(train_loader)} loss {running / i:.4f}")
        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}: test loss {test_loss:.4f}, test accuracy {test_acc * 100:.2f}%")
 
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
 
 
# ---------------------------------------------------------------- inference
def load_model():
    model = DigitCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model
 
 
def preprocess_image(path):
    """Turn any photo/scan of a digit into a 28x28 MNIST-style tensor.
 
    MNIST digits are white on black, centered, ~20px tall inside 28x28.
    """
    img = Image.open(path).convert("L")
    # If the background is light (typical paper/white canvas), invert it.
    if sum(img.tobytes()) / (img.width * img.height) > 127:
        img = ImageOps.invert(img)
    # Crop to the digit's bounding box
    bbox = img.point(lambda p: 255 if p > 50 else 0).getbbox()
    if bbox:
        img = img.crop(bbox)
    # Resize longest side to 20px, keep aspect ratio, paste centered on 28x28
    w, h = img.size
    scale = 20.0 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    return normalize(canvas).unsqueeze(0)  # shape [1, 1, 28, 28]
 
 
def predict(path):
    model = load_model()
    x = preprocess_image(path).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    digit = probs.argmax().item()
    print(f"Predicted digit: {digit}  (confidence {probs[digit] * 100:.1f}%)")
    top3 = probs.topk(3)
    for p, d in zip(top3.values, top3.indices):
        print(f"  {d.item()}: {p.item() * 100:.1f}%")
 
    # Save and open an image: your original next to what the model sees
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(Image.open(path).convert("L"), cmap="gray")
    axes[0].set_title("Your image")
    axes[0].axis("off")
    axes[1].imshow(x.cpu().squeeze(), cmap="gray")
    axes[1].set_title(f"Predicted: {digit} ({probs[digit] * 100:.1f}%)")
    axes[1].axis("off")
    plt.tight_layout()
    import time
    out_name = f"prediction_{time.strftime('%H%M%S')}.png"
    plt.savefig(os.path.abspath(out_name))
    print(f"Saved {out_name}")
    if hasattr(os, "startfile"):
        os.startfile(os.path.abspath(out_name))
 
 
def predict_multi(path):
    """Find every separate digit in an image, predict each, and save an annotated picture.
 
    Needs: pip install scipy
    Works best when digits are separated by a small gap.
    """
    import os
    import time
    import numpy as np
    from scipy import ndimage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
 
    model = load_model()
    arr = np.array(Image.open(path).convert("L"))
    if arr.mean() > 127:          # light background -> make digits white on black
        arr = 255 - arr
    H, W = arr.shape
 
    # 1) find blobs of ink (dilate a little so broken strokes stay together)
    mask = ndimage.binary_dilation(arr > 100, iterations=2)
    labels, _ = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if (y1 - y0) < 0.03 * H and (x1 - x0) < 0.03 * W:   # ignore tiny specks
            continue
        boxes.append([x0, y0, x1, y1])
 
    # 2) drop frames/borders that completely surround another box
    def contains(a, b):
        return a is not b and a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]
    boxes = [b for b in boxes if not any(contains(b, o) for o in boxes)]
    if not boxes:
        print("No digits found in the image.")
        return
 
    # 3) sort into reading order: top-to-bottom rows, left-to-right inside a row
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
 
    # 4) turn every box into a 28x28 MNIST-style tensor and predict them all at once
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
 
    # 5) print results
    idx = 0
    for r in rows:
        line = " ".join(str(preds[idx + i]) for i in range(len(r)))
        print(line)
        idx += len(r)
    print("\nDetails:")
    for i, (d, c) in enumerate(zip(preds, confs), 1):
        flag = "" if c >= 0.9 else "   <- low confidence"
        print(f"  #{i}: {d}  ({c * 100:.1f}%){flag}")
 
    # 6) save an annotated picture
    fig, ax = plt.subplots(figsize=(6, 6 * H / W + 0.5))
    ax.imshow(arr, cmap="gray")
    for (x0, y0, x1, y1), d, c in zip(ordered, preds, confs):
        color = "lime" if c >= 0.9 else "orange"
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
        ax.text(x0, max(y0 - 3, 0), str(d), color=color, fontsize=14, fontweight="bold", va="bottom")
    ax.axis("off")
    plt.tight_layout()
    out_name = f"multi_{time.strftime('%H%M%S')}.png"
    plt.savefig(os.path.abspath(out_name))
    print(f"\nSaved {out_name}")
    if hasattr(os, "startfile"):
        os.startfile(os.path.abspath(out_name))
 
 
def show_samples(n=12):
    import matplotlib.pyplot as plt
    model = load_model()
    test_set = datasets.MNIST("data", train=False, download=True, transform=normalize)
    idxs = torch.randperm(len(test_set))[:n]
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows))
    for ax, idx in zip(axes.flat, idxs):
        x, y = test_set[idx.item()]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(DEVICE)).argmax(1).item()
        ax.imshow(x.squeeze(), cmap="gray")
        ax.set_title(f"pred {pred} / true {y}", color="green" if pred == y else "red")
        ax.axis("off")
    plt.tight_layout()
    import os
    import time
    out_name = f"samples_{time.strftime('%H%M%S')}.png"
    plt.savefig(os.path.abspath(out_name))
    print(f"Saved {out_name}")
    if hasattr(os, "startfile"):
        os.startfile(os.path.abspath(out_name))
 
 
# ---------------------------------------------------------------- main
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "predict" and len(sys.argv) > 2:
        predict(sys.argv[2])
    elif cmd == "multi" and len(sys.argv) > 2:
        predict_multi(sys.argv[2])
    elif cmd == "samples":
        show_samples()
    else:
        print(__doc__)