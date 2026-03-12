from medmnist import PathMNIST, INFO
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

# 1) Load dataset
transform = transforms.ToTensor()
train_dataset = PathMNIST(split="train", download=True, transform=transform)

print("Number of training samples:", len(train_dataset))
sample_img, sample_lbl = train_dataset[0]
print("Example item shape:", sample_img.shape)   # e.g., torch.Size([3, 28, 28])
print("Example label (index):", int(sample_lbl))

# Optional: class names
labels = INFO["pathmnist"]["label"]   # dict: { "0": "adipose", ... }
print("Class names mapping (index -> name):")
for k, v in labels.items():
    print(f"{k}: {v}")

# 2) Show first 5 images
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    img, lbl = train_dataset[i]               # img: torch.Tensor [C,H,W]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img_np = img.permute(1, 2, 0).numpy() # -> [H,W,C]
        if img_np.shape[2] == 1:              # single-channel -> grayscale
            axes[i].imshow(img_np[:, :, 0], cmap="gray")
        else:
            axes[i].imshow(img_np)
    else:
        # fallback for unexpected shapes
        axes[i].imshow(img.squeeze().numpy(), cmap="gray")
    axes[i].set_title(f"Label: {int(lbl)}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()
