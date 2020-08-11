from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W).astype(np.float64)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      temp_dW = np.zeros_like(W)
      scores = X[i].dot(W)
      scores -= np.max(scores)
      exp_scores = np.exp(scores)
      prob_scores = exp_scores/np.sum(exp_scores)  
      for j in range(num_classes):
            if j == y[i]:
                temp_dW[:,j] = (prob_scores[j] - 1) * X[i].reshape(1,W.shape[0])
                continue
            temp_dW[:,j] = (prob_scores[j]) * X[i].reshape(1,W.shape[0])
      dW += temp_dW
      loss += -np.log(prob_scores[y[i]])

    loss /= num_train
    loss += 0.5 * reg * np.sum(W**2)
    
    dW /= num_train
    dW += reg * W

    return loss, dW





def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    def comp_loss_for_example(i):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        prob_scores = exp_scores/np.sum(exp_scores)              
        return -np.log(prob_scores[y[i]])

    f = np.vectorize(comp_loss_for_example)
    loss = np.sum(f(np.arange(num_train)))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W**2)
    
    
    def comp_grad_for_example(i):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        prob_scores = exp_scores/np.sum(exp_scores)
        def update_grad_for_class(j):
              temp_dW = np.zeros_like(W)
              if j == y[i]:
                  temp_dW[:,j] = (prob_scores[j] - 1) * X[i].reshape(1,W.shape[0])
              else:    
                  temp_dW[:,j] = (prob_scores[j]) * X[i].reshape(1,W.shape[0])
              return temp_dW
        f_class = np.vectorize(update_grad_for_class, signature='()->(n,m)')
        temp_dW = np.sum(f_class(np.arange(num_classes)),axis =0)
        
        return temp_dW
    f = np.vectorize(comp_grad_for_example, signature='()->(n,m)')
    dW = np.sum(f(np.arange(num_train)), axis=0)/num_train
    dW += reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
