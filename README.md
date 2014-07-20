# Machine learning with Torch

This repository aims to be a collection of simple machine learning algorithms for [Torch7](http://torch.ch/).

## Regression with MLP

Usually [*multilayer perceptrons*](http://en.wikipedia.org/wiki/Multilayer_perceptron), (*MLPs*), are used for *pattern recognition* (a *classification* task) in the fields of *image* and *speech* recognition. Nevertheless, they can be effectively used for *regression*. Check out the [`MLP-regression`](MLP-regression) section to find out more about it.

## PCA / KLT
[Principal component analysis](http://en.wikipedia.org/wiki/Principal_component_analysis), (*PCA*), or [Karhunen–Loève transform](http://en.wikipedia.org/wiki/Karhunen%E2%80%93Lo%C3%A8ve_theorem), (*KLT*), allows us to smartly reduce the dimensionality of a *data-space*. It can be used for removing the redundancy from input data (and, therefore, speeding up the learning process) and for visualisation purposes (going from, say, 10 dimensions to 3D, which we can better understand). More details can be found in the [`PCA`](PCA) section.
