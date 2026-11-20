**Training behaviour.** Training MSE fell from about 0.056 after epoch 1 to about **0.014** after epoch 20. The validation MSE (noisy test images against clean test images) followed it closely and ended at about **0.0146**. The small gap between the two curves shows the model is not overfitting. Over 20 epochs it generalises well to digits it never saw. Both curves were still falling slowly at epoch 20, so more epochs would bring a small further improvement.

**Denoising quality.** With σ = 0.5, the corruption is severe. The noisy inputs have an average MSE of about **0.116** against the clean images (PSNR ≈ 9.4 dB), and the digit is hard to make out in many of them. The DAE brings this down to about **0.0146** (PSNR ≈ 18.7 dB). That is an **≈ 87 % drop in error** and a gain of about **+9.4 dB**. Every one of the 10 000 test images is closer to its clean version after passing through the DAE than before. In the 3 × 5 figure, the background noise is removed almost completely and each digit's shape and identity are recovered.

**Limitations of the reconstructions.** The outputs are slightly **blurrier** than the originals, and stroke edges are soft. There are two reasons:
1. The MSE loss rewards predicting the *average* of the plausible clean images, which smooths out fine detail.
2. The 128-unit bottleneck cannot keep every pixel-level detail. It keeps the structure that is typical of digits and discards the rest, which is exactly what removes the noise.

The worst cases show the trade-off clearly. These are unusual handwriting styles: a 6 with a very open loop, 8s with odd shapes or a long tail, and a crooked 5. The DAE "corrects" them towards a more typical digit shape, and the reconstruction can even look like a different digit (for example, the 6 looks like a 0). Per digit, **1** is easiest (MSE ≈ 0.007) because it is a simple stroke with few lit pixels. **8** and **2** are hardest (≈ 0.018) because they have the most curved, varied strokes.

**Robustness to other noise levels.** The model was trained only with σ = 0.5, but it still works across a range:
- For σ between 0.2 and 1.0 the reconstruction is always far better than the noisy input. Even at σ = 1.0, the DAE MSE (≈ 0.040) is lower than the noisy input's MSE at σ = 0.3.
- The error grows smoothly as σ increases, so performance degrades gradually rather than failing suddenly.
- At very low noise (σ = 0.1), the DAE output (≈ 0.0099) is *worse* than the barely-noisy input (≈ 0.0054). The network always applies its learned smoothing, and its reconstruction error has a floor of about 0.01. At that noise level, the blur it adds costs more than the noise it removes.

This suggests training on a range of noise levels if the noise strength is not known in advance.

**Learned representation.** A 2-D PCA projection of the 128-d bottleneck codes, coloured by the true labels, shows clear structure even though the labels were never used in training:
- **1s** form a tight, separate cluster.
- **0s**, and **7s / 9s / 4s**, sit in distinct regions.
- Digits with similar shapes overlap (4 with 9, 3 with 5 and 8).

The first two components explain only about 20 % of the variance, so the full 128-d code holds much more separation than this 2-D view shows. About 39 % of the bottleneck activations are exactly zero because of ReLU, so the code is fairly **sparse**.

**Conclusion.** The symmetric 784-256-128-256-784 DAE, trained with Adam and MSE for 20 epochs, learns a robust, compact representation of handwritten digits. It removes heavy Gaussian noise (σ = 0.5), cutting reconstruction error by about 87 %, and it generalises to unseen images and to other noise levels. The main weakness is a mild blur, which comes from the MSE objective and the bottleneck. Ways to reduce it include convolutional layers, a perceptual or binary cross-entropy loss, or training on a range of noise levels.
