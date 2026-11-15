---
title: Datasets
nav_order: 3
---

# Datasets
{: .no_toc }

Two public datasets with different content, resolution, and generator families.
{: .lead }

| Property          | CIFAKE | RajuFace / 140k Real and Fake Faces |
| ----------------- | ------ | ----------------------------------- |
| Source            | [Kaggle dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | [Hugging Face dataset](https://huggingface.co/datasets/RajuFace/140k-Real-and-Fake-Faces) |
| Images            | 120,000: 60,000 REAL + 60,000 FAKE | 140,000: 70,000 REAL + 70,000 FAKE |
| Native resolution | 32x32 | 256x256 |
| REAL source       | CIFAR-10 photographs; diverse object categories | FFHQ face photographs |
| FAKE source       | Stable Diffusion 1.4 (diffusion) | StyleGAN (GAN) |
| Original splits   | Train 100,000; test 20,000 | Train 100,000; validation 20,000; test 20,000 |
| Split proportions | Train/test = 5:1 | Train/validation/test = 5:1:1 (71.43% / 14.29% / 14.29%) |

Both pipelines use **REAL = 0, FAKE = 1** and preserve the original partitions.

**CIFAKE** established the small-image baseline. Its 32x32 resolution limits fine-detail
analysis, and wavelet boundary effects are substantial at that size.

**RajuFace** checks the same features on higher-resolution faces and a different generator
family. It is loaded by streaming from Hugging Face at a pinned dataset revision; the full
dataset is not downloaded. RajuFace's `generator` field is kept as metadata only and never
enters the feature extractor.

{: .note }
The two datasets differ in content, resolution, sources, and generation method, so they do
not allow a controlled comparison of diffusion and GAN artifacts.

## Why not GenImage?

Our original plan used a small, balanced subset of the GenImage benchmark [[1]](references.md).
After multiple attempts, we were unable to reassemble its split ZIP archive into a usable file,
so we switched to CIFAKE. See [Progress](progress.md) for how the dataset choice evolved.
