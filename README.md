# Vit-on-small-data
## 🚀 Introduction
In this repo, we present the lightest Vision Transformer (ViT) trained from scratch to achieve ~93% accuracy on CIFAR-10 within just 50 epochs.  
To the best of our knowledge, this is the highest accuracy ever reported with the **lowest compute** and **fastest training time** in the literature for ViTs trained from scratch on CIFAR-10.

---
## 📝 Abstract:
Vision Transformers (ViTs) are among the most impactful architectural innovations in computer vision, demonstrating that autoregressive architectures can be applied to image data while achieving state-of-the-art results. However, ViTs are notoriously data-hungry — their performance scales strongly with dataset size, making them less effective on smaller benchmarks. To address this limitation, we conducted extensive ablation studies to isolate the core inefficiencies of ViTs on small datasets and propose a minimal yet effective modification. Our approach enables ViTs to achieve competitive accuracy on CIFAR-10 with a significantly reduced computational footprint and faster convergence, offering a lightweight solution for resource-constrained training scenarios.
