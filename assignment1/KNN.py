import numpy as np
import random
from cs231n.data_utils import  load_CIFAR10
import matplotlib.pyplot as plt
import os

class KNearestNeighbor:  # 首先是定义一个处理KNN的类
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]#测试样本数
        num_train = self.X_train.shape[0]#训练样本数
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                #两层循环
                dists[i,j]=np.sqrt(np.dot(X[i]-self.X_train[j],X[i]-self.X_train[j]))

                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            dists[i,:]=np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis=1))

            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        sq_train=np.sum(np.square(self.X_train),axis=1)#(5000,)
        sq_test=np.sum(np.square(X),axis=1) #(500,)
        sq_train=np.reshape(sq_train,(1,num_train))
        sq_test=np.reshape(sq_test,(num_test,1))
        mul=np.multiply(np.dot(X,self.X_train.T),-2)#(500,5000)
        dists_1=sq_train+mul
        dists=sq_test+dists_1
        dists=np.sqrt(dists)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # training point, and use self.y_train to find the labels of these      #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            sort=np.argsort(dists[i,:])#按降序排列
            index=sort[0:k]#取前k个距离最小的
            index=list(index)
            closest_y=self.y_train[index]
            closest_y=np.ravel(closest_y)
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            y_pred[i] = np.argmax(np.bincount(closest_y))
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred




#load the raw CIFAR-10 data
os.chdir('E://Python//deep learning CS231n//assignment1')
cifar10_dir='E://Python//deep learning CS231n//assignment1//cs231n//datasets'
X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)
print('Training data shape:',X_train.shape)
print("Training labels shape:",y_train.shape)
print('Test data shape:',X_test.shape)
print('Test labels shape:',y_test.shape)

#we show a few examples of training images from each classes
'''classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes=len(classes)
samples_per_class=7
for y,cls in enumerate(classes):
    idxs=np.flatnonzero(y_train==y)#记录y_train中等于y的索引值
    idxs=np.random.choice(idxs,samples_per_class,replace=False)#选出7张图
    for i,idx in enumerate(idxs):
        plt_idx=i* num_classes+y+1
        plt.subplot(samples_per_class,num_classes,plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()'''
#调整数据集的大小
num_training=5000
mask=range(num_training)
X_train=X_train[mask]
y_train=y_train[mask]

num_test=500
mask=range(num_test)
X_test=X_test[mask]
y_test=y_test[mask]
#把所有图片变成一列
X_train=np.reshape(X_train,(X_train.shape[0],-1))
X_test=np.reshape(X_test,(X_test.shape[0],-1))
print (X_train.shape,X_test.shape)





#用KNN进行训练
classifier=KNearestNeighbor()
classifier.train(X_train,y_train)
'''
#用两层循环计算
dists=classifier.compute_distances_two_loops(X_test)
print (dists.shape)

y_test_pred=classifier.predict_labels(dists,k=1)
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print("Got %d/ %d correct => accuracy: %f" %(num_correct,num_test,accuracy))
'''
'''
#计算一层循环的结果
dists_one=classifier.compute_distances_one_loop(X_test)

#检查两次距离是否一样
difference=np.linalg.norm(dists-dists_one,ord=2)
print("Difference was: %f" % difference)
if difference<0.001:
    print('Good! the distance matricecs are the same')
else:
    print("Uh-oh! the distance matrices are different")

#full-vectorized version
dists_two=classifier.compute_distances_no_loops(X_test)
#检查距离是否一样
difference = np.linalg.norm(dists - dists_two, ord='fro')
print ('Difference was: %f' % (difference, ))
if difference < 0.001:
  print ('Good! The distance matrices are the same')
else:
  print ('Uh-oh! The distance matrices are different')
'''
#检查所用时间长短
def time_function(f,*args):
    import time
    tic=time.time()
    f(*args)
    toc=time.time()
    return toc-tic

#two_loop_time=time_function(classifier.compute_distances_two_loops,X_test)
#print("Two loop version took %f seconds" %two_loop_time)

#one_loop_time=time_function(classifier.compute_distances_one_loops,X_test)
#print("One loop version took %f seconds" %one_loop_time)

no_loop_time=time_function(classifier.compute_distances_no_loops,X_test)
print("No loop version took %f seconds" %no_loop_time)


#筛选不同的k
num_folds=5
k_choices=[1,3,5,8,10,12,15,20,50,100]

X_train_folds=[]
y_train_folds=[]
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds=np.array_split(X_train,num_folds)#分成num_folds份验证集, list数据类型
y_train_folds=np.array_split(y_train,num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    k_to_accuracies[k]=np.zeros(num_folds)

    for i in range(num_folds):
        Xtr=np.array(X_train_folds[:i]+X_train_folds[i+1:])
        ytr=np.array(y_train_folds[:i]+y_train_folds[i+1:])
        Xte=np.array(X_train_folds[i])
        yte=np.array(y_train_folds[i])

        Xtr=np.reshape(Xtr,(np.int32(X_train.shape[0] * 4 / 5), -1))
        ytr = np.reshape(ytr, (np.int32(y_train.shape[0] * 4 / 5), -1))
        Xte=np.reshape(Xte,(np.int32(X_train.shape[0]/5),-1))
        yte = np.reshape(yte, (np.int32(y_train.shape[0] / 5), -1))

        classifier.train(Xtr,ytr)
        yte_pred=classifier.predict(Xte,k)
        yte_pred=np.reshape(yte_pred,(yte_pred.shape[0],-1))
        num_correct=np.sum(yte_pred==yte)
        accuracy=float(num_correct)/len(yte)
        k_to_accuracies[k][i]=accuracy


#print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print ('k = %d, accuracy = %f' % (k, accuracy))



