# PCA
*Principal component analysis* (*PCA*) finds the directions of greatest variance in a dataset.

## How does it work?
For finding the first component, PCA looks for a (linear) combination of the elements of the vectors of your dataset, which explains the highest variation of the data. Think of it as the *versor* on which, the projection of the dataset (*dot product*-ed) will have its highest variability.
For the second component, PCA looks for another (linear) combination — i.e. another versor — orthogonal to the first one, which explains the second highest variation of the data.
For the third, same story. In this case, the (linear) combination has to be vertical to all previously found one. Etc...

## What is used for?
For each *versor*, *principal component* or *eigenvector* there is an associated *power* (or *energy*, if square rooted), *variance* or *eigenvalue* which tell us the "amount of variability". What happens often is that only the first few components have non-neglectable variance. Hence, data dimensionality can be greatly reduced with little loss of information.
In turn, dimensionality reduction and knowledge of variance distribution can be used to

 - speed up training
 - 2/3D representation of data living in *n*-D, *n* > 3
 - data augmentation by aligned perturbation
 - ZCA whitening

About ZCA, it's worth mentioning that sphering data does not always produce better results (see [Feldman's blog post](http://blog.explainmydata.com/2012/07/should-you-apply-pca-to-your-data.html), for example).
