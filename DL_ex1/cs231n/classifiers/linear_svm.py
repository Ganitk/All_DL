from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    for i in range(num_train):
        temp_dW = np.zeros(W.shape)
        count_margn_non_zero = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                count_margn_non_zero += 1
                temp_dW[:,j] = X[i].reshape(1,W.shape[0])
        temp_dW[:,y[i]] = -X[i] * count_margn_non_zero
        dW += temp_dW
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * np.sum(2*W)

   
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    correct_class_score = scores[np.arange(num_train),y].reshape(num_train,1)
    margin = scores - correct_class_score + 1
    margin = np.maximum(0, scores - correct_class_score + 1)
    margin[ np.arange(num_train), y] = 0 # do not consider correct class in loss
    loss = margin.sum() 
    non_zero_margin = (margin > 0).sum(axis=1)
    margin[margin > 0] = 1
    margin[np.arange(num_train),y ] -= non_zero_margin
    dW = (X.T).dot(margin) 
    
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * np.sum(2*W)




    # def comp_for_example(i):
    #     scores = (X[i].dot(W)).astype(np.float64)
    #     correct_class_score = scores[y[i]]
    #     def comp_for_class(j):
    #           if j == y[i]:
    #               return 0 
    #           margin = scores[j] - correct_class_score + 1 # note delta = 1
    #           if margin > 0:
    #               return margin
    #           else:
    #               return 0
    #     comp_for_class_func = np.vectorize(comp_for_class)
    #     r = np.arange(num_classes)
    #     return comp_for_class_func(r)
        

    # comp_for_example_func = np.vectorize(comp_for_example,signature='()->(n)')
    # r = comp_for_example_func(list(range(num_train)))
    # loss = np.sum(r)
    # # Right now the loss is a sum over all training examples, but we want it
    # # to be an average instead so we divide by num_train.
    # loss /= num_train
    # # Add regularization to the loss.
    # loss += reg * np.sum(W * W)
    # # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # #############################################################################
    # # TODO:                                                                     #
    # # Implement a vectorized version of the gradient for the structured SVM     #
    # # loss, storing the result in dW.                                           #
    # #                                                                           #
    # # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # # to reuse some of the intermediate values that you used to compute the     #
    # # loss.                                                                     #
    # #############################################################################
    # # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # def comp_for_example(i):
    #     scores = X[i].dot(W)
    #     temp_dW = np.zeros(W.shape)
    #     correct_class_score = scores[y[i]]
    #     def comp_for_class(j):
    #           if j == y[i]:
    #               return 0 
    #           margin = scores[j] - correct_class_score + 1 # note delta = 1
    #           if margin > 0:
    #               temp_dW[:,j] = X[i].reshape(1,W.shape[0])
    #               return 1
    #           else:
    #               return 0
    #     comp_for_class_func = np.vectorize(comp_for_class)
    #     r = list(range(num_classes))
    #     non_zero_margin = np.sum(comp_for_class_func(r))
    #     temp_dW[:,y[i]] = -X[i] * non_zero_margin
    #     return temp_dW

    # comp_for_example_func = np.vectorize(comp_for_example, signature='()->(n,m)')
    # r = comp_for_example_func(list(range(num_train)))
    # dW = np.sum(r, axis=0)
    # dW /= num_train
    # dW += reg * np.sum(2*W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
