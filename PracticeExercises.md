## 1. How Many Parameters in Network?

Consider a 2-layered network.
- Layer 1: Convolutional with `kernel_size` $5 * 5$, `padding` = 1, `stride` = 2
- Layer 2: Fully-connected
- `in_channels` = 3
- `hidden_channels` = 7
- `out_feat` = 2
- `input_size` of image = $h * w = 12 * 12$

**What is the total no. of learnable parameters?**

CNN:
- We have 3 kernels for each input channel and 7 hidden channels, so $3 * 5 * 5 * 7 = 525$ weights
- Each output channel has 1 bias, so in total $7$ bias parameters
- Total: $525 + 7 = 532$

FCN:
- First we calculate the new input size after CNN using $Hp = Wp = 1 + (H + 2 * padding - K) // stride = 1 + (12 + 2 * 1 - 5) // 2 = 5$
- So the FCN receives as input images of $5 * 5$ and 7 channels, which gets flattened
- The weight matrix has $5 * 5 * 7 = 175$ rows and $2$ columns 
- Since the no. of output features is 2, we have $175 * 2 = 350$ weight parameters and 2 biases
- Total: $350 + 2 = 352$

The total learnable parameters for all layers is $532 + 352 = 884$


## 2. Computing MLE for Poisson Distribution

Given the Poisson distribution:

$$ P_{model}(X = x) = \frac{\lambda^xe^{-\lambda}}{x!} $$

We draw $m$ samples: {$x_1, x_2, ..., x_m$}. The objective is to find the MLE for parameter $\lambda$ as a function of data samples.

**Write down the likelihood function for the given probability distribution.**

This is simply the product of the PDF for the observed values.

$$ L(\lambda, x_1, ..., x_m) = \prod_{j=1}^m{\frac{\lambda^x_je^{-\lambda}}{x_j!}} $$



**Write the log-likelihood function by using the logarithm operator on the function obtained in previous step.**

Hint: the log-likelihood is defined as:

$$ MLE(X) = argmax_{\theta} \prod_{1=1}^m{log(P_{model}(x_i, \theta))} $$

where $X$ = {${x_1...x_m}$}

$$ L(\lambda, x_1, ..., x_m) = ln(\prod_{j=1}^m{\frac{\lambda^x_je^{-\lambda}}{x_j!}}) $$

$$ = \sum_{j=1}^m{ln(\frac{\lambda^x_je^{-\lambda}}{x_j!})} $$

$$ = \sum_{j=1}^m{[x_j ln(\lambda) - \lambda - ln(x_j!)]} $$

$$ = -m\lambda + ln(\lambda)\sum_{j=1}^m{x_j} - \sum_{j=1}^m{ln(x_j!)} $$

**Calculate the derivative of the natural log likelihood function with respect to $\lambda$.**

$$ \frac{d}{d\lambda} L(\lambda, x_1, ..., x_m) $$

$$ = \frac{d}{d\lambda} ({-m\lambda + ln(\lambda)\sum_{j=1}^m{x_j} - \sum_{j=1}^m{ln(x_j!)} } ) $$

$$ = -m + \frac{1}{\lambda} \sum_{j=1}^m{x_j} $$

**Set the derivative equal to 0 and solve for $\lambda$.**

$$ -m + \frac{1}{\lambda} \sum_{j=1}^m{x_j} = 0 $$

$$ \lambda_{ML} = \frac{1}{m} \sum_{j=1}^m{x_j} $$



## 3. Receptive Field in CNN


## 4. Batch Normalization


## 5. Regularization



## 6. LTSMs



## 7. Self-Attention



## 8. Contractive Auto-encoder