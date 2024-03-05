# TUD-CS4240-Deep-Learning

## 1 Activations and Backpropagation

### `Class Linear(in_features, out_features)`

__*PyTorch version: `nn.Linear`*__

- Fully-connected layer given by $y = x^T W + b$
- For a linear layer with $N_{in}$ and $N_{out}$ neurons, there are $N_{in} * N_{out}$ weights (connections) and $N_{out}$ biases
    - Input $x \in \mathbb{R}^{N_{in}}$, or typically in batches: $x \in \mathbb{R}^{batch, N_{in}}$
    - Output $y \in \mathbb{R}^{N_{out}}$ 
    - Weight matrix $W \in \mathbb{R}^{N_{in} * N_{out}}$ 

```python
    def init_params(self, std=1.):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias uniform distribution.
        
        Args:
            std: Standard deviation of Gaussian distribution (default: 1.0)
        """
```

```python
    def forward(self, x):
        """
        Forward pass of linear layer: multiply input tensor by weights and add
        bias. Store input tensor as cache variable.
        
        Args:
            x: input tensor

        Returns:
            y: output tensor
        """
```

```python
    def backward(self, dupstream):
        """
        Backward pass of linear layer: calculate gradients of loss with respect
        to weight and bias and return downstream gradient dx.
        
        Args:
            dupstream: Gradient of loss with respect to output of this layer.

        Returns:
            dx: Gradient of loss with respect to input of this layer.
        """
```


### Non-linear activation functions: `ReLU(x)` and `Sigmoid(x)`
- Similarly to regular linear layers, implement `def forward(x)` and `def backward(dupstream)`


### `Class Net(layers)`

__*PyTorch version: `TorchNet(in_features, hidden_dim, out_features)`*__
- *Does not take in layers, but initalizes those*
- *`nn.Parameter` can be used to load parameters from our model*

```python
 def reset_params(self, std=1.):
        """
        Reset network parameters. Applies `init_params` to all layers with
        learnable parameters.
        
        Args:
            std: Standard deviation of Gaussian distribution (default: 0.1)
        """
```

```python
def forward(self, x):
        """
        Performs forward pass through all layers of the network.
        
        Args:
            x: input tensor

        Returns:
            x: output tensor
        """
```

```python
def backward(self, dupstream):
        """
        Performs backward pass through all layers of the network.
        
        Args:
            dupstream: Gradient of loss with respect to output.
        """
```

```python
def optimizer_step(self, lr):
        """
        Updates network weights by performing a step in the negative gradient
        direction in each layer. The step size is determined by the learning
        rate.
        
        Args:
            lr: Learning rate to use for update step.
        """
```



## 2.1 CNNs

-  CNNs have far fewer parameters compared to fully-connected networks, which is beneficial for image data for instance: in which the individual pixels are the input neurons.
    - Rather than treating pixels that are far apart and close-by equally, we wanna have a more clever way of embedding the spatial structure! 

- Decrease in image size $\leftrightarrow$ increase in kernel size i.e. larger **receptive field (RF)** for neurons
    - Convolution increases RF linearly 
    - Pooling increases RF multiplicatively  

### `class Conv2d(in_channels, out_channels, kernel_size, stride, padding)`

__PyTorch version: `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`__

```python
    def init_params(self, std=0.7071):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias will be zeros.
        
        Args:
            std: Standard deviation of Gaussian distribution (default: 0.7071)
        """
```

``` python
    def forward(self, x):
        """
        Forward pass of convolutional layer
        
        Args:
            x: input tensor which has a shape of (N, C, H, W)

        Returns:
            y: output tensor which has a shape of (N, F, H', W') where
                H' = 1 + (H + 2 * padding - kernel_size) / stride
                W' = 1 + (W + 2 * padding - kernel_size) / stride
        """
```

```python
    def backward(self, dupstream):
        """
        Backward pass of convolutional layer: calculate gradients of loss with
        respect to weight and bias and return downstream gradient dx.
        
        Args:
            dupstream: Gradient of loss with respect to output of this layer.

        Returns:
            dx: Gradient of loss with respect to input of this layer.
        """
```


### `class MaxPool2d(kernel_size, stride, padding)`

__PyTorch version: `nn.MaxPool2d(kernel_size, stride, padding)`__

```python
 def forward(self, x):
        """
        Forward pass of max pooling layer
        
        Args:
            x: input tensor with shape of (N, C, H, W)
 
        Returns:
            y: output tensor with shape of (N, C, H', W') where
                H' = 1 + (H + 2 * padding - kernel_size) / stride
                W' = 1 + (W + 2 * padding - kernel_size) / stride
        """
```

```python
    def backward(self, dupstream):
        """
        Backward pass of max pooling layer: calculate gradients of loss with
        respect to weight and bias and return downstream gradient dx.
        
        Args:
            dupstream: Gradient of loss with respect to output of this layer.
 
        Returns:
            dx: Gradient of loss with respect to input of this layer.
        """
```


### `class CNN(in_channels, hidden_channels, out_features)`

__PyTorch version: `nn.TorchCNN(in_channels, hidden_channels, out_features)`__

```python
def forward(self, x):
    """
    Forward pass of CNN: chains inputs and outputs
    """
```


## 2.2 Optimization Algorithms

- **Stochastic Gradient Descent (SGD)** tends to be noisy! We need a memory-efficient way of computing gradients based on past statistics (averaging leads to smoother trajectories)!

- **EWMA: Exponentially Weighted Moving Average**: due to recursion, we only need to store $2n$ values given $n$ derivative values - one for each previous value.
    - $S_t = \rho S_{t-1} + (1 - \rho) y_t$ 

Some Gradient Descent Update algorithms that make use of EWMA:
- **Momentum**: on average in good direction - uses average to update gradient directly
- **RMSProp**: high variance in wrong direction - uses average to scale learning rate of update such that we take larger steps towards beginning
- **Adam**: combines **Momentum** and **RMSProp**


## 3.1 Regularization

**Overfitting** occurs when there is a big gap between *true error* and *apparent error* - likely caused by a small training set hindering generalization of the model

Ways to beat overfitting:
- Data augmentation 
- Reduce the no. of features
- Reduce complexity/flexibility of the model

This is where **regularization** comes in: it discourages overly complex models!

- *"Regularization are techniques to reduce the test error, possibly at the expense of increased training error"*
- **Weight decay** discourages large weights
- **Ridge (L2)**: shrinks coefficients of less significant features towards 0
- **Lasso (L1)**: reduce weights to exactly 0 (induces 
sparsity), performs feature selection during training NN
- **Early stopping** of gradient updates is useful when network is correctly initialized 
    - i.e. we stop training when discrepancy between validation and apparent error starts to increase: decreasing error on train set may make it seem like model is learning still, whereas validation set (unseen) error increasing...
    - This ensures norm of `w` is not too large (starting with small weights is good)
    - Better generalization to future data
- **Noise robustness**
    - Noise added to input - data augmentation
    - Noise added to weights - encourages stability of model
    - Noise added to output - label smoothing 
- **Dropout** = combination of weight decay and noise injection
    - Each training step: randomly select a fraction of the nodes and make then inactive (set to 0) 
    - Prevents NN to become overly reliant on specific neurons

## 3.2 RNNs


## 4.1 Self-Attention


## 4.2 Unsupervised