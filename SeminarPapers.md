## [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341)

Four troubling trends:
- *"Failure to distinguish between Explanation vs Speculation"*
  - There is not sufficient explanation for some claims
  - Intuitions phrased as factual statements
- *"Failure to identify the sources of empirical gains"*
  - Gains come from hyper-parameter tuning instead of the proposed method (unnecessary modifications to neural architectures)
  - Other sources: new model, data preprocessing, etc.
- *Mathiness*
  - The use of mathematics that obfuscates or impresses rather than clarifies, e.g. by confusing technical and non-technical concepts
  - Lacking cohesion between text and maths
- *Misuse of language*
  - e.g. by choosing terms of art with colloquial connotations or by overloading established technical terms



## [A Step Toward Quantifying Independently Reproducible Machine Learning Research](https://arxiv.org/abs/1909.06674)

- Releasing code is very important to reproduce ML research


## [Do ImageNet Classifiers Generalize to ImageNet?](https://arxiv.org/abs/1902.10811)

- **Main motivation: re-use of the test set leads to overfitting**
  - They build new test sets for CIFAR-10 and ImageNet to test to which extent current classification models generalize to new data
- **So, do they generalize? No, because the technical term "generalization" only applies to the original data**
  - The accuracy drops are not caused by adaptivity, but by the models' inability to generalize to slightly "harder" images than those found in the original test sets.



## [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431)



## [The Low-Rank Simplicity Bias in Deep Networks](https://arxiv.org/abs/2103.10427 )




## [Scaling down Deep Learning](https://arxiv.org/abs/2011.14439)

- **MNIST dataset is scaled down**, which has the advantage of accelerating the iteration cycle of eploratory research
- We introduce MNIST-1D: a minimalist, low-memory, and low-compute alternative to classic deep learning benchmarks
- Issues with original MNIST dataset:
  - It does a poor job of differentiating between linear, non-linear, and translation-invariant models
  -	It is somewhat large for a toy dataset (784-dimensional vector)
  -	It is hard to hack/adapt
  


## [Group normalization](https://arxiv.org/abs/1803.08494)

- The statistics are computed over groups of pixels
- The grouping is done by grouping activations


## [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html)



## [What Algorithms can Transformers Learn? A Study in Length Generalization](https://arxiv.org/abs/2310.16028 )




## [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

- Idea: cycling between domains to ...

