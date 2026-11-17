---
title: Reproducibility
nav_order: 9
has_children: true
has_toc: false
---

# Reproducibility
{: .no_toc }

How to get the datasets.
{: .lead }

## 140k Real and Fake Faces (main dataset)

The dataset is public on Hugging Face:
[RajuFace/140k-Real-and-Fake-Faces](https://huggingface.co/datasets/RajuFace/140k-Real-and-Fake-Faces).
It is stored as Parquet files (a table format), each holding about 10,000 JPEG images and
their labels (0 = real, 1 = fake). We use two files, about 290 MB each:

| Role | File |
| --- | --- |
| Training | `data/validation-00000-of-00002.parquet` |
| Testing | `data/test-00000-of-00002.parquet` |

Download them from the dataset's **Files and versions** tab, or from a terminal. We pin the
dataset version (`1f99c8e8...`) so the files cannot change under us:

```bash
mkdir -p datasets/140k
REV=1f99c8e831601e3a300dd3de670df23384a9be6c
BASE=https://huggingface.co/datasets/RajuFace/140k-Real-and-Fake-Faces/resolve/$REV/data
curl -L -o datasets/140k/validation-00000-of-00002.parquet $BASE/validation-00000-of-00002.parquet
curl -L -o datasets/140k/test-00000-of-00002.parquet $BASE/test-00000-of-00002.parquet
```

Place the files in `datasets/140k/` in the repository. This folder is not uploaded to GitHub.

## CIFAKE (method check)

Download CIFAKE from
[Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(a free Kaggle account is required) and extract it so the folders are
`datasets/CIFAKE/train/REAL`, `datasets/CIFAKE/train/FAKE`, `datasets/CIFAKE/test/REAL`, and
`datasets/CIFAKE/test/FAKE`.

## Code

All code is in the [GitHub repository](https://github.com/eecs351-team/EECS351-project). The
pages below give more technical detail for the first experiments.
