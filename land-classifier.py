# 1. Install the datasets library (run once)
# pip install datasets

from datasets import load_dataset

# 2. Load the Agriculture-Vision dataset
#    This will download and cache the data locally.
dataset = load_dataset("shi-labs/Agriculture-Vision")

# 3. Inspect the splits
print(dataset)  
# e.g. {'train': Dataset(num_rows=15000, ...), 'validation': ..., 'test': ...}

# 4. Access an example
example = dataset["train"][0]
# The example dict contains:
#  - 'image' : PIL.Image of shape (512, 512, 3) or (512, 512, 4) if NIR included
#  - 'mask'  : segmentation mask (NumPy array) with anomaly labels

print("Image size:", example["image"].size)
print("Mask shape:", example["mask"].shape)

# 5. (Optional) Visualize with matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(example["image"])
axes[0].set_title("RGB Image")
axes[0].axis("off")

axes[1].imshow(example["mask"], cmap="tab20")
axes[1].set_title("Anomaly Mask")
axes[1].axis("off")

plt.show()
