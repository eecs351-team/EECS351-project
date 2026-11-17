---
title: Classification
nav_order: 5
---

# Classification
{: .no_toc }

Do the frequency features actually detect generated faces?
{: .lead }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Setup

- **Data.** 140k Real and Fake Faces: 5,000 REAL + 5,000 FAKE training images and a separate
  1,000 REAL + 1,000 FAKE test set from a different file of the dataset (see
  [Datasets](datasets.md)). REAL = 0, FAKE = 1.
- **Fair evaluation.** Every setting (feature scaling and each classifier's parameters) is
  chosen with 5-fold cross-validation on the training images only. The test images are used
  once, at the end.
- **Chance level.** The classes are balanced, so guessing gives 50% accuracy.

## Logistic Regression

Logistic Regression estimates the probability that an image is FAKE from a weighted sum of
its features:

$$
P(\text{FAKE} \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}} .
$$

Each weight shows how much a feature pushes an image toward FAKE.

### Basic features, compared by transform

As planned in our proposal, we first compare the 11 basic features grouped by transform.

| Features | Number of features | Test accuracy |
| --- | --- | --- |
| Fast Fourier Transform energy shares | 3 | 53.8% |
| Discrete Cosine Transform energy shares | 3 | 52.9% |
| Discrete Wavelet Transform energy shares | 5 | 54.1% |
| All 11 together | 11 | 55.5% |

{% include figure.html src="/docs/figures/faces/classification_basic.png" alt="Bar chart of test accuracy for each transform's features, and the confusion matrix of the 11-feature model" caption="**Figure 1.** Left: test accuracy of Logistic Regression for each transform. The dashed line is chance (50%). Right: confusion matrix for all 11 features; rows are the true class and columns the predicted class." %}

No transform stands out, and all are only a little better than guessing. Three wide frequency
bands are too coarse to describe the small differences between real and generated faces.

### Finer frequency features

Keeping more of the spectrum's shape helps (feature definitions are on the
[Methods](methods/index.md) page):

| Features | Number of features | Test accuracy |
| --- | --- | --- |
| All 11 basic features | 11 | 55.5% |
| Radial Fourier spectrum | 127 | 60.0% |
| Radial Fourier spectrum with a Hann window | 127 | 63.2% |
| Cosine blocks | 1,024 | 65.0% |
| Region features | 32 | 59.5% |
| **All combined** | 1,194 | **68.2%** |
| Control: JPEG compression setting only | 1 | 50.0% |

The control row shows that both classes were compressed identically, so the classifier is not
using a compression shortcut.

We also asked whether the classifier uses the **overall amount** of fine detail or the
**shape** of the spectrum (how energy is spread across frequencies). Shape alone keeps the full
63.2%; the overall amount alone gives only 55.1%. The useful information is in the shape.

## Comparing classifiers

We then trained five more classifiers on the same features:

- **Linear Support Vector Machine (SVM):** a straight-line boundary, like Logistic Regression.
- **Radial basis function (RBF) SVM:** allows a curved boundary.
- **K-Nearest Neighbors:** labels an image like the most similar training images.
- **Gaussian Naive Bayes:** treats each feature as independent.
- **Decision Tree:** a series of yes/no threshold questions.

For the 1,194 combined features, all classifiers first reduce them to 100 summary directions
(principal component analysis, PCA) so that every classifier gets the same input.

| Classifier | 11 basic features | Radial spectrum, Hann window | All combined |
| --- | --- | --- | --- |
| Logistic Regression | 55.6% | 63.2% | 68.8% |
| Linear SVM | 55.5% | 63.2% | 68.9% |
| **RBF SVM** | **56.5%** | **66.1%** | **74.2%** |
| K-Nearest Neighbors | 54.5% | 61.5% | 58.4% |
| Gaussian Naive Bayes | 54.0% | 55.2% | 59.3% |
| Decision Tree | 52.7% | 58.6% | 61.1% |

{% include figure.html src="/docs/figures/task3/classifiers.png" alt="Heat map of test accuracy for six classifiers and three feature sets, and the confusion matrix of the best model" caption="**Figure 2.** Left: test accuracy for every classifier and feature set; darker is better. Right: confusion matrix of the best model, the RBF SVM on all combined features (713 of 1,000 real and 771 of 1,000 fake faces classified correctly)." %}

**What we learned:**

- **Features matter more than the classifier.** Better features raise Logistic Regression from
  55% to 69%; the best classifier adds about 5 more points.
- **The RBF SVM is best on every feature set**, reaching **74.2%** (area under the receiver
  operating characteristic curve, AUC, 0.81; 0.5 is chance and 1 is perfect).
- **Simpler models do worse.** Naive Bayes, K-Nearest Neighbors, and the Decision Tree all fall
  below Logistic Regression. Our features are many, correlated, and individually weak, which
  suits none of these three.
- With 2,000 test images, each accuracy is uncertain by about ±2 percentage points, so
  differences of one or two points are not meaningful.

## Evaluation metrics

| Metric | What it tells us |
| --- | --- |
| Accuracy | Fraction of all test images classified correctly |
| Precision | Of the images predicted FAKE, how many are FAKE |
| Recall | Of the FAKE images, how many are detected |
| F1 score | Balance of precision and recall |
| AUC | Probability that a random fake image gets a higher "fake" score than a random real one |
| Confusion matrix | Counts of each kind of correct and wrong prediction |

Full details, including every feature group and the chosen settings, are on the
[Task 2](../docs/task2.md) and [Task 3](../docs/task3.md) pages under Progress.

All metrics for every model are in the result files:
[Logistic Regression results (CSV)](../docs/results/faces_metrics.csv) and
[classifier comparison results (CSV)](../docs/results/task3_metrics.csv).
