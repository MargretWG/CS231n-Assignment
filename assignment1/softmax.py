import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.softmax import softmax_loss_naive,softmax_loss_vectorized


def get_CIFAR10_Data(num_train=9000,num_validation=1000,num_test=1000,num_dev=500):
    cifar10_dir='cs231n//datasets'
    X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)
    #subsample the data
    mask=range(num_train,num_train+num_validation)
    X_val=X_train[mask]
    y_val=y_train[mask]
    mask=range(num_train)
    X_train=X_train[mask]
    y_train=y_train[mask]
    mask=range(num_test)
    X_test=X_test[mask]
    y_test=y_test[mask]
    mask=np.random.choice(num_train,num_dev,replace=False)#从0-num_train中选num_dev个样本
    X_dev=X_train[mask]
    y_dev=y_train[mask]

    #reshape the image data into rows
    X_train=np.reshape(X_train,(X_train.shape[0],-1))
    X_val=np.reshape(X_val,(X_val.shape[0],-1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))


    #normalize the data: substract the mean image
    mean_image=np.mean(X_train,axis=0)#按列取平均，得到9000幅图片的均值图片
    X_train-=mean_image
    X_val-=mean_image
    X_test-=mean_image
    X_dev-=mean_image

    #add bias dimension and transoform into columns
    X_train=np.hstack([X_train,np.ones((X_train.shape[0],1))])
    X_val=np.hstack([X_val,np.ones((X_val.shape[0],1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])


    return X_train,y_train,X_val,y_val,X_test,y_test,X_dev,y_dev

#Invoke the above function to get our data
X_train,y_train,X_val,y_val,X_test,y_test,X_dev,y_dev=get_CIFAR10_Data()
print('Train data shape:', X_train.shape)
print('Train labels shape:', y_train.shape)
print('Validation data shape:', X_val.shape)
print('Validation labels shape:', y_val.shape)
print('Test data shape:', X_test.shape)
print('Test labels shape:', y_test.shape)
print('dev data shape:', X_dev.shape)
print('dev labels shape:', y_dev.shape)

#Classification
import time
#Generate a random softmax weight matrix and use it to compute the loss
W=np.random.randn(3073,10)*0.0001
loss,grad=softmax_loss_naive(W,X_dev,y_dev,0.0)
#大约loss应该是接近-log(0.1),此处为合理性检查
print ('loss:%f' % loss)
print('sanity check: %f'%(-np.log(0.1)))

#梯度检查
from cs231n.gradient_check import grad_check_sparse
f=lambda w: softmax_loss_naive(w,X_dev,y_dev,0.0)[0]
grad_numerical=grad_check_sparse(f,W,grad,10)

#加上正则化项，再做一遍梯度检查
loss,grad=softmax_loss_naive(W,X_dev,y_dev,1e2)
f=lambda w: softmax_loss_naive(w,X_dev,y_dev,1e2)[0]
grad_numerical=grad_check_sparse(f,W,grad,10)


#vectorized version
'''tic=time.time()
loss_naive,grad_naive=softmax_loss_naive(W,X_dev,y_dev,0.00001)
toc=time.time()
print('naive loss: %e computed in %fs'%(loss_naive,toc - tic))

tic=time.time()
loss_vec,grad_vec=softmax_loss_vectorized(W,X_dev,y_dev,0.00001)
toc=time.time()
print('vectorized loss: %e computed in %fs'%(loss_vec,toc-tic))

#compare the two versions of the gradient
grad_difference=np.linalg.norm(grad_naive-grad_vec,ord='fro')
print('Loss differencce: %f'%np.abs(loss_naive-loss_vec))
print('Gradient difference:%f'% grad_difference)'''

#用validation set调整超参数！ 正则化强度和学习率
from cs231n.classifiers import Softmax
result={}
best_val=-1
best_softmax=None
learning_rate=[5e-6,1e-7,5e-7]
reg=[1e4,5e4,1e8]
#################################################
for each_rate in learning_rate:
    for each_reg in reg:
        softmax=Softmax()
        loss_hist=softmax.train(X_train,y_train,learning_rate=each_rate,reg=each_reg,num_iters=700,verbose=True)
        y_train_pred=softmax.predict(X_train)
        accuracy_train=np.mean(y_train==y_train_pred)

        y_val_pred=softmax.predict(X_val)
        accuracy_val=np.mean(y_val==y_val_pred)
        result[each_rate,each_reg]=(accuracy_train,accuracy_val)
        if(best_val<accuracy_val):
            best_val=accuracy_val
            best_softmax=softmax
####################################################
for lr,reg in sorted(result):
    train_accuracy, val_accuracy=result[(lr,reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f'%(lr,reg,train_accuracy,val_accuracy))

print("best validation accuracy achieved druring cross-validation: %f" % best_val)

#evaluate on test set,evaluate the best softmax on test set
y_test_pred=best_softmax.predict(X_test)
test_accuracy=np.mean(y_test_pred==y_test)
print('softmax on raw pixels final test set accuracy：%f ' % test_accuracy)