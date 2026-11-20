"""Builds DAE_MNIST.ipynb (the assignment notebook) from the cells below.

Run:  .venv\\Scripts\\python.exe build_notebook.py
Then execute it with: jupyter nbconvert --to notebook --execute --inplace DAE_MNIST.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Assignment 3: Implementing and Evaluating a Denoising Autoencoder (DAE)

**Name:** Rabail Fatima
**Roll Number:** 23L-2588

**Objective.** Design, implement and evaluate a multi-layer Denoising Autoencoder on MNIST. The network
receives handwritten digits corrupted by Gaussian noise (σ = 0.5) and learns to reconstruct the clean digit.

**Notebook outline**
1. Data preparation: load, normalise to [0, 1], flatten to 784-d, build clean targets and noisy inputs (clamped to [0, 1])
2. Autoencoder architecture: 784 → 256 → **128** → 256 → 784
3. Training: Adam, MSE loss, 20 epochs, batch size 256
4. Results and analysis: the required 3 × 5 image grid (clean / noisy / reconstructed), plus loss curves, quantitative metrics and a noise-robustness check
""")

code(r"""
import os, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
""")

md(r"""
## 1. Data Preparation

* Load MNIST (60 000 training and 10 000 test images, 28 × 28 greyscale).
* Normalise pixel values from [0, 255] to **[0, 1]** by dividing by 255.
* Flatten each 28 × 28 image into a **784-dimensional** vector.
* **Clean data** = normalised images (the *target*).
* **Noisy data** = clean data + Gaussian noise with mean 0 and **standard deviation 0.5** (the *input*), then
  **clamped to [0, 1]** with `np.clip`.

The same corruption is applied to the test set so the model can be evaluated on unseen noisy digits.
""")

code(r"""
# Load MNIST (labels are not needed for an autoencoder)
(x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Raw shapes :", x_train.shape, x_test.shape, "| dtype:", x_train.dtype,
      "| range:", x_train.min(), "-", x_train.max())

# Normalise to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Flatten 28x28 -> 784
x_train = x_train.reshape((len(x_train), 784))
x_test  = x_test.reshape((len(x_test), 784))
print("Flattened  :", x_train.shape, x_test.shape)
""")

code(r"""
NOISE_STD = 0.5

def add_gaussian_noise(x, std=NOISE_STD, rng=np.random.default_rng(SEED)):
    # Corrupt x with zero-mean Gaussian noise and clamp the result to [0, 1].
    noisy = x + rng.normal(loc=0.0, scale=std, size=x.shape).astype("float32")
    return np.clip(noisy, 0.0, 1.0)

# Clean data = target, noisy data = input
x_train_clean, x_test_clean = x_train, x_test
x_train_noisy = add_gaussian_noise(x_train_clean)
x_test_noisy  = add_gaussian_noise(x_test_clean)

for name, arr in [("x_train_clean", x_train_clean), ("x_train_noisy", x_train_noisy),
                  ("x_test_clean", x_test_clean),   ("x_test_noisy", x_test_noisy)]:
    print(f"{name:14s} shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")

assert x_train_noisy.min() >= 0.0 and x_train_noisy.max() <= 1.0, "noisy data must be clamped to [0, 1]"
assert x_test_noisy.min() >= 0.0 and x_test_noisy.max() <= 1.0
""")

code(r"""
# Quick look at the training pairs the network will see (noisy input -> clean target)
fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for i in range(8):
    axes[0, i].imshow(x_train_clean[i].reshape(28, 28), cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(x_train_noisy[i].reshape(28, 28), cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_title("Clean (target)", loc="left", fontsize=10)
axes[1, 0].set_title("Noisy, σ = 0.5 (input)", loc="left", fontsize=10)
plt.tight_layout(); plt.show()
""")

md(r"""
## 2. Autoencoder Architecture

A symmetric, fully-connected DAE:

| Part | Layer | Units | Activation |
|---|---|---|---|
| Input | Input | 784 | – |
| Encoder | Dense 1 | 256 | ReLU |
| Encoder | Dense 2 (bottleneck) | 128 | ReLU |
| Decoder | Dense 3 | 256 | ReLU |
| Decoder | Output | 784 | Sigmoid |

The sigmoid output keeps every reconstructed pixel in (0, 1), matching the range of the clean targets.
The encoder and decoder are also built as separate sub-models so the 128-d latent code can be inspected.
""")

code(r"""
# Encoder
inputs     = keras.Input(shape=(784,), name="noisy_input")
h1         = layers.Dense(256, activation="relu", name="enc_dense_256")(inputs)
bottleneck = layers.Dense(128, activation="relu", name="bottleneck_128")(h1)

# Decoder
h3      = layers.Dense(256, activation="relu", name="dec_dense_256")(bottleneck)
outputs = layers.Dense(784, activation="sigmoid", name="reconstruction_784")(h3)

dae = keras.Model(inputs, outputs, name="denoising_autoencoder")

# Separate encoder / decoder views that share the trained weights
encoder = keras.Model(inputs, bottleneck, name="encoder")
latent_in = keras.Input(shape=(128,), name="latent_input")
decoder = keras.Model(latent_in,
                      dae.get_layer("reconstruction_784")(dae.get_layer("dec_dense_256")(latent_in)),
                      name="decoder")

dae.summary()
""")

md(r"""
## 3. Training

* Optimiser: **Adam** (default learning rate 0.001)
* Loss: **Mean Squared Error** between the reconstruction and the *clean* image
* **20 epochs**, **batch size 256**, data shuffled every epoch
* The noisy test set (with its clean targets) is used as validation data to track generalisation
""")

code(r"""
dae.compile(optimizer="adam", loss="mse")

history = dae.fit(
    x_train_noisy, x_train_clean,          # input = noisy, target = clean
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test_clean),
    verbose=2,
)
""")

code(r"""
epochs = np.arange(1, len(history.history["loss"]) + 1)
plt.figure(figsize=(7, 4))
plt.plot(epochs, history.history["loss"], "o-", label="Training MSE")
plt.plot(epochs, history.history["val_loss"], "s-", label="Validation (test) MSE")
plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("DAE training curve")
plt.xticks(epochs); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150); plt.show()

print(f"Final training MSE  : {history.history['loss'][-1]:.5f}")
print(f"Final validation MSE: {history.history['val_loss'][-1]:.5f}")
""")

md(r"""
## 4. Results and Analysis

### 4.1 Required visualisation: clean vs noisy vs reconstructed

Five different test images (one per column), with three rows:
* **Row 1:** original clean test image
* **Row 2:** noisy input image (σ = 0.5, clamped)
* **Row 3:** DAE reconstructed output
""")

code(r"""
x_test_recon = dae.predict(x_test_noisy, batch_size=256, verbose=0)

n = 5
idx = [0, 1, 2, 3, 4]   # first five test images -> five different digits (7, 2, 1, 0, 4)

row_titles = ["Original (clean)", "Noisy input", "DAE reconstruction"]
fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7))
for col, i in enumerate(idx):
    for row, data in enumerate([x_test_clean, x_test_noisy, x_test_recon]):
        ax = axes[row, col]
        ax.imshow(data[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"Test #{i} (label {y_test[i]})", fontsize=10)
        if col == 0:
            ax.set_ylabel(row_titles[row], fontsize=11)
fig.suptitle("Denoising Autoencoder on MNIST (Gaussian noise σ = 0.5)", fontsize=13)
plt.tight_layout()
plt.savefig("dae_results.png", dpi=150)
plt.show()
""")

md(r"""
### 4.2 Quantitative evaluation on the full test set

MSE and PSNR (peak signal-to-noise ratio, peak = 1) against the clean images, for the noisy input and for the DAE output.
Lower MSE and higher PSNR are better.
""")

code(r"""
def per_image_mse(a, b):
    return np.mean((a - b) ** 2, axis=1)

def psnr(mse, peak=1.0):
    return 10 * np.log10(peak ** 2 / mse)

mse_noisy = per_image_mse(x_test_noisy, x_test_clean)
mse_recon = per_image_mse(x_test_recon, x_test_clean)

print(f"{'':22s}{'MSE':>10s}{'PSNR (dB)':>12s}")
print(f"{'Noisy input':22s}{mse_noisy.mean():10.5f}{psnr(mse_noisy).mean():12.2f}")
print(f"{'DAE reconstruction':22s}{mse_recon.mean():10.5f}{psnr(mse_recon).mean():12.2f}")
print(f"\nMSE reduction: {100 * (1 - mse_recon.mean() / mse_noisy.mean()):.1f}%")
print(f"PSNR gain    : {psnr(mse_recon).mean() - psnr(mse_noisy).mean():+.2f} dB")
print(f"Test images improved by the DAE: {np.mean(mse_recon < mse_noisy) * 100:.2f}%")

print("\nPer-digit reconstruction MSE:")
for d in range(10):
    m = y_test == d
    print(f"  digit {d}: {mse_recon[m].mean():.5f}")
""")

code(r"""
# Hardest test images for the DAE (largest reconstruction error)
worst = np.argsort(mse_recon)[-5:][::-1]
fig, axes = plt.subplots(3, 5, figsize=(11, 7))
for col, i in enumerate(worst):
    for row, data in enumerate([x_test_clean, x_test_noisy, x_test_recon]):
        ax = axes[row, col]
        ax.imshow(data[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(f"#{i} label {y_test[i]}\nMSE {mse_recon[i]:.3f}", fontsize=9)
        if col == 0: ax.set_ylabel(row_titles[row], fontsize=10)
fig.suptitle("Five worst reconstructions on the test set", fontsize=12)
plt.tight_layout(); plt.savefig("worst_cases.png", dpi=150); plt.show()
""")

md(r"""
### 4.3 Robustness to other noise levels

The DAE was trained only with σ = 0.5. Here the test set is corrupted with different noise levels to see how the
reconstruction quality degrades.
""")

code(r"""
levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
rng_eval = np.random.default_rng(123)
noisy_mse, recon_mse = [], []
for s in levels:
    xn = np.clip(x_test_clean + rng_eval.normal(0, s, x_test_clean.shape).astype("float32"), 0, 1)
    xr = dae.predict(xn, batch_size=512, verbose=0)
    noisy_mse.append(per_image_mse(xn, x_test_clean).mean())
    recon_mse.append(per_image_mse(xr, x_test_clean).mean())
    print(f"σ = {s:.1f}:  noisy MSE = {noisy_mse[-1]:.5f}   DAE MSE = {recon_mse[-1]:.5f}")

plt.figure(figsize=(7, 4))
plt.plot(levels, noisy_mse, "o-", label="Noisy input")
plt.plot(levels, recon_mse, "s-", label="DAE reconstruction")
plt.axvline(0.5, color="gray", ls="--", lw=1, label="Training σ = 0.5")
plt.xlabel("Gaussian noise σ at test time"); plt.ylabel("MSE vs clean image")
plt.title("Noise robustness of the DAE"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("noise_robustness.png", dpi=150); plt.show()
""")

md(r"""
### 4.4 Latent space (128-d bottleneck)

The encoder maps each noisy image to a 128-d code. A 2-D PCA projection of these codes, coloured by the true digit
label (never seen in training), shows whether the bottleneck has learned a meaningful representation.
""")

code(r"""
codes = encoder.predict(x_test_noisy, batch_size=512, verbose=0)
print("Latent code shape:", codes.shape,
      f"| fraction of inactive (zero) units: {np.mean(codes == 0):.2%}")

# PCA with plain NumPy (SVD)
c = codes - codes.mean(axis=0)
_, S, Vt = np.linalg.svd(c, full_matrices=False)
proj = c @ Vt[:2].T
explained = (S ** 2) / np.sum(S ** 2)

plt.figure(figsize=(7, 6))
sc = plt.scatter(proj[:, 0], proj[:, 1], c=y_test, cmap="tab10", s=3, alpha=0.6)
plt.colorbar(sc, ticks=range(10), label="Digit label")
plt.xlabel(f"PC1 ({explained[0]:.1%} var)"); plt.ylabel(f"PC2 ({explained[1]:.1%} var)")
plt.title("PCA of the 128-d bottleneck codes (noisy test images)")
plt.tight_layout(); plt.savefig("latent_pca.png", dpi=150); plt.show()
""")

md(r"""
### 4.5 Discussion

__ANALYSIS__
""")

code(r"""
# Save the trained model for later use
dae.save("dae_mnist.keras")
print("Saved model to dae_mnist.keras")
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

import sys
analysis = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else \
    "*(Discussion is written after the notebook has been executed.)*"
for c in nb["cells"]:
    if c["cell_type"] == "markdown" and "__ANALYSIS__" in c["source"]:
        c["source"] = c["source"].replace("__ANALYSIS__", analysis.strip())

nbf.write(nb, "DAE_MNIST.ipynb")
print("Wrote DAE_MNIST.ipynb with", len(cells), "cells")
