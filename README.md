# EECS 351 Project: AI-Generated Image Detection
Ethan Kocheril, Jiayuan Zhu, Zhirui Wang, Ziyi Xie
## Overview
As the development of large language models continues to progress, the danger of generated images being
used for fraud, impersonation and other malicious ends will become more present.
Our project focuses on implementing ways to discover generated images through a combination of traditional
DSP techniques like the Discrete Cosine Transform and machine-learning models like the support vector machine
and logistic regression classifier. We are currently using images from the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), which is a dataset
comprised of 32x32 images, both real and ai-generated.

## Methods
At the moment, we have just computed the FFT, DWT, and DCT of real and generated images. 
insert plots here
