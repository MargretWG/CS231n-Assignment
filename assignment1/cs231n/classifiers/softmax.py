import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):#带循环
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
  dW = np.zeros_like(W)
  dW_each=np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_class=W.shape[1]
  f=np.dot(X,W)  #(N,C)，评分函数
  f_max=np.reshape(np.max(f,axis=1),(num_train,1)) #找到每一行的最大值，然后reshape 之后减去
  #这样可以防止后面的操作会出现数值上的一些偏差
  #regularization
  f-=f_max
  p = np.exp(f) / np.sum(np.exp(f),axis=1,keepdims=True) #N by C #这里要注意，除的是每个样本的和，不能全求和
  #求交叉熵！！！
  loss=0.0
  y_true=np.zeros_like(p)
  y_true[np.arange(num_train),y]=1.0# 生成hot-vector
  for i in range(num_train):
    for j in range(num_class):
      loss+=-(y_true[i,j]*np.log(p[i,j])) #损失函数公式：L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
      dW_each[:,j]=-(y_true[i,j]-p[i,j])*X[i,:] # ∇Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk, where Pk = exp(fk)/∑j exp(fj
    dW+=dW_each
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W) #加上正则项
  dW/=num_train
  dW+=reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):#向量化操作
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)#D by C

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  f = np.dot(X, W)  # (N,C)，评分函数
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))  # 找到每一行的最大值，然后reshape 之后减去
  # 这样可以防止后面的操作会出现数值上的一些偏差
  # regularization
  f -= f_max
  p = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)  # N by C #这里要注意，除的是每个样本的和，不能全求和
  # 求交叉熵！！
  y_true=np.zeros_like(p)
  y_true[np.arange(num_train),y]=1.0# 生成hot-vector
  loss+=-np.sum(np.log(p[np.arange(num_train),y])) / num_train + 0.5* reg*np.sum(W*W)
  dW+=-np.dot(X.T,y_true-p) /num_train + reg* W ##求梯度的vectorized 形式

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

