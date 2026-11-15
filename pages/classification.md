---
title: Classification
nav_order: 5
---

# Classification
{: .no_toc }

Testing whether the DSP features are useful for detection.
{: .lead }

{: .planned }
No classifier has been trained or evaluated yet. This page describes the plan and will hold
the classification results once they exist.

## Logistic Regression

We start with binomial Logistic Regression on the 11 DSP features, with REAL = 0 and
FAKE = 1. The model estimates

$$
P(\text{FAKE} \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}} ,
$$

so each learned weight shows how much a feature pushes an image toward FAKE. Feature
scaling will be fitted on training data only, and the model will be evaluated on held-out
images that were not used during training.

## Other classifiers

If time allows, we will compare classifiers on the same extracted features, starting with an
**SVM** and possibly **K-Nearest Neighbors**, **Naive Bayes**, and **Decision Trees**. This shows
whether a more complex classifier actually improves detection.

## Evaluation

| Metric | What it tells us |
| --- | --- |
| Accuracy | Overall fraction of correct predictions |
| Precision | Of the images predicted FAKE, how many are FAKE |
| Recall | Of the FAKE images, how many are detected |
| F1 score | Balance of precision and recall |
| Confusion matrix | Where the errors are |

We also plan to compare feature groups (FFT only, DCT only, DWT only, all features) to see which
transform contributes most, and to test generalization to unseen data and, if time permits,
sensitivity to compression and resizing.

## Results

<!-- Add accuracy / precision / recall / F1, the confusion matrix figure, and the
     feature-group comparison here once the classifier is trained. -->

*To be added.*
