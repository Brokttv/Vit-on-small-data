# Vit-on-small-data
##  Introduction
In this repo, we present the lightest Vision Transformer (ViT) trained from scratch to achieve ~93% accuracy on CIFAR-10 within just 50 epochs.  
To the best of our knowledge, this is the highest accuracy ever reported with the **lowest compute** and **fastest training time** in the literature for ViTs trained from scratch on CIFAR-10.

---
## Abstract:
Vision Transformers (ViTs) are among the most impactful architectural innovations in computer vision, demonstrating that autoregressive architectures can be applied to image data while achieving state-of-the-art results. However, ViTs are notoriously data-hungry — their performance scales strongly with dataset size, making them less effective on smaller benchmarks. To address this limitation, we conducted extensive ablation studies to isolate the core inefficiencies of ViTs on small datasets and propose a minimal yet effective modification. Our approach enables ViTs to achieve competitive accuracy on CIFAR-10 with a significantly reduced computational footprint and faster convergence, offering a lightweight solution for resource-constrained training scenarios.

---
##  Key Ablation:


As widely recognized, the primary reason behind Vision Transformers (ViTs) underperforming on small datasets is their lack of *instant inductive bias*. ViTs gradually accumulate meaningful feature representations but typically require large datasets and prolonged training to do so effectively. 

In contrast, Convolutional Neural Networks (CNNs) excel at extracting high-level features early in training due to their convolutional layers, which inherently encode spatial inductive bias.

To address this gap, we replaced the raw linear patchification step in vanilla ViTs with a **3-block CNN-based patch embedding**. This small change introduces inductive bias from the very first layer, resulting in **faster convergence** and **better accuracy** on small datasets such as CIFAR-10.

<br>

##  Experiment Configuration

| **Component**          | **Value** |
|------------------------|-----------|
| **Dataset**            | CIFAR-10 |
| **Image Size**         | 32×32 |
| **Augmentations**      | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `TrivialAugmentWide`, Normalize (mean = `[0.4914, 0.4822, 0.4465]`, std = `[0.2470, 0.2435, 0.2616]`) |
| **Batch Size**         | 32 |
| **Num Workers**        | 2 |
| **Pin Memory**         | `True` |
| **Patch Embedding**    | CNN-based projection to `embedding_dim=192` |
| **Transformer Layers** | `2` |
| **Attention Heads**    | `3` |
| **Attention Dropout**  | `0.0` |
| **MLP Size**           | `1152` |
| **MLP Dropout**        | `0.1` |
| **Embedding Dropout**  | `0.1` |
| **Optimizer**          | AdamW (`lr=2e-3`, `weight_decay=0.001`, `betas=(0.9, 0.999)`, `eps=1e-8`) |
| **Scheduler**          | Cosine Decay with Warmup (`warmup_steps=20000`, `total_steps=78125`) |
| **Mixup**              | `alpha=0.1` (λ ~ Beta(α,α)), applied to every training batch via `mixup_data(x, y, alpha=0.1, device='cuda')`. Returns `(mixed_x, y_a, y_b, lam)` and uses a random permutation for pairing. |
| **Loss Function**      | `CrossEntropyLoss` combined with `mixup_criterion(loss_fn, pred, y_a, y_b, lam)` |
| **Gradient Clipping**  | `clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| **Random Seed**        | `42` |


## Model Comparison

| Model            | Params (M) | FLOPs  | Accuracy (%) | Epochs | source |
|------------------|-----------|-----------|-------------|--------|-------|
| **Our ViT (CNN-Patchified)** | **2.3**     | *300M*    | **93.32** | **50** | Trained from scratch with CNN patch embed |
| CCT-14/7x2       | 3.76       | 2.38G    | 96.53        | 300    | [paper](https://arxiv.org/pdf/2104.05704)|
| Add-ViT           | 26.8       | *Fill*    | 94.97        | 300    | [paper](https://link.springer.com/article/10.1007/s11063-024-11643-8) |
 |
| Vanilla ViT-Tiny | 5.7       | *Fill*    | ~78.0       | 300    | Poor performance without pretraining |
| DeiT-Tiny        | 5.7       | *Fill*    | 79.8        | 300    | Requires strong augmentations + distillation |

> **Key Takeaway:** Our model achieves competitive accuracy with **lowest parameter count and compute cost** among ViT-like models trained from scratch on CIFAR-10 within only 50 epochs.



