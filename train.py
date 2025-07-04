import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import label
from skimage import morphology

# Parameters
DATA_ROOT = Path("/content/drive/MyDrive/Data/Geri_Imaging")
DOWNSAMPLE_SCALE = 0.1
LINE_LENGTH_THRESHOLD = 10  # Lowered to detect more fine lines
KERNEL_SIZE = (5, 5)        # Smaller kernel for tighter fitting
DILATE_ITERATIONS = 2       # Slightly reduced to preserve detail

# Step 1: Find first TIF
tif_paths = sorted(DATA_ROOT.rglob("*.tif"))
if not tif_paths:
    raise FileNotFoundError(f"No .tif files found under {DATA_ROOT}")
tif_path = tif_paths[0]
print("Full TIFF:", tif_path.name)

# Step 2: Load full image
with rasterio.open(tif_path) as src:
    full_img = src.read(1).astype(np.float32)

# Step 3: Normalize and downsample
full_img = (full_img - full_img.min()) / (full_img.max() - full_img.min() + 1e-8)
H, W = full_img.shape
Hs, Ws = int(H * DOWNSAMPLE_SCALE), int(W * DOWNSAMPLE_SCALE)
small = cv2.resize(full_img, (Ws, Hs), interpolation=cv2.INTER_AREA)
small8 = (small * 255).astype(np.uint8)

# Step 4: Generate mirrored versions
augmented_imgs = {
    "original": small8,
    "flip_lr": cv2.flip(small8, 1),
    "flip_ud": cv2.flip(small8, 0),
    "flip_both": cv2.flip(small8, -1)
}

# Step 5: LSD + region contour drawing
fig, axes = plt.subplots(1, len(augmented_imgs), figsize=(5 * len(augmented_imgs), 5))

for ax, (name, img_aug) in zip(axes, augmented_imgs.items()):
    # LSD detection
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    raw, _, _, _ = lsd.detect(img_aug)
    lines = raw[:, 0, :] if raw is not None else np.zeros((0, 4))

    # Create binary mask of lines
    line_mask = np.zeros_like(img_aug, dtype=np.uint8)
    for x0, y0, x1, y1 in lines.astype(int):
        length = np.hypot(x1 - x0, y1 - y0)
        if length >= LINE_LENGTH_THRESHOLD:
            cv2.line(line_mask, (x0, y0), (x1, y1), 255, 1)

    # Dilate lines to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
    dilated = cv2.dilate(line_mask, kernel, iterations=DILATE_ITERATIONS)

    # Invert to get land areas
    land_mask = cv2.bitwise_not(dilated)

    # Clean up noise and holes
    cleaned = morphology.remove_small_holes(land_mask > 0, area_threshold=200).astype(np.uint8) * 255
    cleaned = morphology.remove_small_objects(cleaned > 0, min_size=200).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Show image and overlay contours
    ax.imshow(img_aug, cmap='gray', origin='upper')
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 2 and len(cnt) > 4:
            ax.plot(cnt[:, 0], cnt[:, 1], color='lime', linewidth=1)

    ax.set_title(name)
    ax.axis('off')

plt.suptitle(f"LSD-Derived Land Regions (Fine Edges)", fontsize=14)
plt.tight_layout()
plt.show()
