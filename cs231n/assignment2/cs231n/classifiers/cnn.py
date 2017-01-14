import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - [optional: conv - relu] - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,
               use_batchnorm=False, extra_conv=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.extra_conv = extra_conv
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
    self.params['b1'] = np.zeros(shape=num_filters)
    
    if self.use_batchnorm:
        self.params['g1'] = np.ones(shape=num_filters)
        self.params['B1'] = np.zeros(shape=num_filters)
        self.params['g2'] = np.ones(shape=1)
        self.params['B2'] = np.zeros(shape=1)
        self.bn_params = [{'mode': 'train'} for i in xrange(2)]
    
    if extra_conv:
        self.params['W1c'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
        self.params['b1c'] = np.zeros(shape=num_filters)
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
    
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(num_filters*input_dim[1]/2*input_dim[2]/2,hidden_dim))
    self.params['b2'] = np.zeros(shape=hidden_dim)
    
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(shape=num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    use_batchnorm = self.use_batchnorm
    extra_conv = self.extra_conv
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    if use_batchnorm:
        g1 = self.params['g1']
        B1 = self.params['B1'] 
        g2 = self.params['g2'] 
        B2 = self.params['B2']
        for bn_param in self.bn_params:
            bn_param[mode] = mode
    
    if extra_conv:
        W1c,b1c = self.params['W1c'], self.params['b1c']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1c = X
    if extra_conv:
        out1c, cache1c = conv_relu_forward(X, W1c, b1c, conv_param)
    out1, cache1 = conv_relu_pool_forward(out1c, W1, b1, conv_param, pool_param)
    if use_batchnorm:
        outb1, cacheb1 = spatial_batchnorm_forward(out1,g1,B1,self.bn_params[0])
    else:
        outb1 = out1
    out2, cache2 = affine_relu_forward(outb1, W2, b2)
    if use_batchnorm:
        outb2, cacheb2 = batchnorm_forward(out2,g2,B2,self.bn_params[1])
    else:
        outb2 = out2
    out3, cache3 = affine_forward(outb2, W3, b3)
    scores = out3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dloss = softmax_loss(scores, y)
    loss += .5 * self.reg * np.sum([np.sum(self.params['W1']**2), np.sum(self.params['W2']**2), np.sum(self.params['W3']**2)])
    
    dx3, dw3, db3 = affine_backward(dloss, cache3)
    if use_batchnorm:
        dxb2, grads['g2'],grads['B2'] = batchnorm_backward_alt(dx3,cacheb2)
    else:
        dxb2 = dx3
    dx2, dw2, db2 = affine_relu_backward(dxb2, cache2)
    if use_batchnorm:
        dxb1, grads['g1'],grads['B1'] = spatial_batchnorm_backward(dx2,cacheb1)
    else:
        dxb1 = dx2
    dx1, dw1, db1 = conv_relu_pool_backward(dxb1, cache1)
    if extra_conv:
        dx1c, dwlc, db1c =  conv_relu_backward(dx1, cache1c)
    
    grads['W1'],grads['b1'] = dw1+ self.reg*self.params['W1'], db1
    grads['W2'],grads['b2'] = dw2+ self.reg*self.params['W2'], db2
    grads['W3'],grads['b3'] = dw3+ self.reg*self.params['W3'], db3
    
    if extra_conv:
        grads['W1c'],grads['b1c'] = dwlc + self.reg*self.params['W1c'], db1c
        
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

pass