import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    special_j, special_count = 0,0
    for j in xrange(num_classes):
      if j == y[i]:
        special_j = j
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW.T[j] += X[i]
        special_count+=1
        loss += margin
    dW.T[special_j] += -special_count*X[i]
    special_count = 0
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += .5*reg * np.sum(W * W)
  dW += 2*.5*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  scores = X.dot(W) #C,N
  correct_class = scores[np.arange(N),y] #N,1
  margin = scores.T - correct_class+ 1 #.reshape(len(correct_class),1) + 1 #broadcast across C 
  margin[y,np.arange(N)] = 0 #set correct class margin to zip (as some student pointed out in lecture, '-= 1' should work too) 
  mod_margin = np.ceil(margin.clip(0,1)) #idenity-ify the raw values (zeros and ones MAT) and mult by X to compute partial deriv
  #above feels kinda hacky (not a numpy guru). none the less, does what I want
  #import pdb; pdb.set_trace()
  mod_margin[y, np.arange(N)] = -mod_margin.sum(axis=0)#sum across rows, the other partial deriv coeff unique for correct class scores
  #mod_margin now has correct scalars for X to compute partial deriv at each i,j
  grad = X.T.dot(mod_margin.T)/N #sum up all partial deriv coeff * X and take avg by dividing by number summed
  #Note: I am currently terrible with thinking through dimmensions (to .T or not to .T). Hope to get better(faster) at this intuition-wise
  loss = margin.clip(min=0).sum()/N
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
 

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dW = grad #expletive! I debugged forever to realize I forgot to assign to dW (feel like PBS Arthur has something to say about this) 
  # unimpressed with my matrix munip skills.. (ex: margin shape is transpose of score)
  dW += 2*.5*reg*W #always add reg penalty to your gradient kids!
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return loss, dW  
