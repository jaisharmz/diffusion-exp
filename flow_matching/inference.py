import os
import time
import matplotlib.pyplot as plt

def get_timestamp():
    return int(time.time() * 1e3)

def make_plot(x, x_recon, filename="eval"):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(x_recon, cmap='gray')
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{get_timestamp()}_{filename}.png")
    plt.close(fig)

def plot_one(x, filename="img"):
    fig = plt.figure()
    plt.imshow(x, cmap='gray')
    plt.axis("off")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{get_timestamp()}_{filename}.png")
    plt.close(fig)