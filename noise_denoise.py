import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load image and convert to grayscale
def load_image(path):
    img = Image.open(path).convert('L')  # 'L' mode = grayscale
    return np.array(img)

# Save NumPy array back to image
def save_image(array, filename):
    img = Image.fromarray(np.uint8(array))
    img.save(filename)

# Add random noise
def add_noise(img_array, noise_level=30):
    noise = np.random.randint(-noise_level, noise_level, img_array.shape)
    noisy_img = img_array + noise
    return np.clip(noisy_img, 0, 255)

# Simple denoising with averaging filter
def denoise_image(img_array, kernel_size=3):
    padded_img = np.pad(img_array, pad_width=kernel_size//2, mode='edge')
    denoised = np.zeros_like(img_array)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            denoised[i, j] = np.mean(region)

    return np.clip(denoised, 0, 255)

def main():
    if not os.path.exists("input_image.jpg"):
        print("Image not found! Place 'input_image.jpg' in the project folder.")
        return

    # Step 1: Load image
    img = load_image("input_image.jpg")

    # Step 2: Add noise
    noisy = add_noise(img)
    save_image(noisy, "noisy_image.jpg")

    # Step 3: Denoise
    denoised = denoise_image(noisy)
    save_image(denoised, "denoised_image.jpg")

    # Step 4: Show all images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original")
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title("Noisy")
    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title("Denoised")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
