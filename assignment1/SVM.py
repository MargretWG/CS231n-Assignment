'''In this exercise you will:\n",
    "- implement a fully-vectorized **loss function** for the SVM\n",
    "- implement the fully-vectorized expression for its **analytic gradient**\n",
    "- **check your implementation** using numerical gradient\n",
    "- use a validation set to **tune the learning rate and regularization** strength\n",
    "- **optimize** the loss function with **SGD**\n",
    "- **visualize** the final learned weights\n'''

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

#读入数据
def GetData():
    #读入数据
    cifar10_dir="cs231n//datasets"
    X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)
    #As a sanity check, we print out the size of the training and test data

    print('Training data shape:',X_train.shape)
    print("Training labels shape:",y_train.shape)
    print("Test data shape:",X_test.shape)
    print("Test labels shape:",y_test.shape)

    #Split the data into train, val, test sets. 还有额外的一个development set作为Training set 的子集
    num_training=9000
    num_validation=1000
    num_test=1000
    num_dev=500

    mask=range(num_training,num_training+num_validation)
    X_val=X_train[mask]
    y_val=y_train[mask]
    mask=range(num_training)
    X_train=X_train[mask]
    y_train=y_train[mask]

    mask=np.random.choice(num_training,num_dev,replace=False)
    X_dev=X_train[mask]
    y_dev=y_train[mask]

    mask=range(num_test)
    X_test=X_test[mask]
    y_test=y_test[mask]
    print("===============================================")
    print ('Train data shape: ', X_train.shape)
    print ('Train labels shape: ', y_train.shape)
    print ('Validation data shape: ', X_val.shape)
    print( 'Validation labels shape: ', y_val.shape)
    print ('Test data shape: ', X_test.shape)
    print ('Test labels shape: ', y_test.shape)

    #将图像转化为向量
    X_train=np.reshape(X_train,[X_train.shape[0],-1])
    X_test=np.reshape(X_test,(X_test.shape[0],-1))
    X_val=np.reshape(X_val,[X_val.shape[0],-1])
    X_dev=np.reshape(X_dev,[X_dev.shape[0],-1])
    print("===============================================")
    print ('Train data shape: ', X_train.shape)
    print ('Train labels shape: ', y_train.shape)
    print ('Validation data shape: ', X_val.shape)
    print( 'Validation labels shape: ', y_val.shape)
    print ('Test data shape: ', X_test.shape)
    print ('Test labels shape: ', y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

X_train,y_train,X_val,y_val,X_test,y_test,X_dev,y_dev=GetData()
#减去均值图像
mean_image=np.mean(X_train,axis=0)
print(mean_image[:10])
'''plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))
plt.show()'''#可视化
X_train-=mean_image
X_test-=mean_image
X_val-=mean_image
X_dev-=mean_image

#加上偏差biases

X_train=np.hstack([X_train,np.ones((X_train.shape[0],1))])
X_test=np.hstack([X_test,np.ones((X_test.shape[0],1))])
X_val=np.hstack([X_val,np.ones((X_val.shape[0],1))])
X_dev=np.hstack([X_dev,np.ones((X_dev.shape[0],1))])

print(X_train.shape,"\n",X_test.shape,"\n",X_val.shape,"\n",X_dev.shape)


#SVM分类器
from cs231n.classifiers.linear_svm import svm_loss_naive
import time
#generate a random SVM weight matrix of small numbers

W=np.random.randn(3073,10)*0.0001
start=time.time()
loss,grad=svm_loss_naive(W,X_dev,y_dev,0.00001)
end=time.time()
last1=end-start
print("loss: %f, time lasted:%fs"% (loss,last1))

#梯度检查,不带正则化项
from cs231n.gradient_check import grad_check_sparse
f=lambda w: svm_loss_naive(w,X_dev,y_dev,0.0)[0]
grad_numerical=grad_check_sparse(f,W,grad)

#梯度检查，带正则化项
loss,grad=svm_loss_naive(W,X_dev,y_dev,1e2)
f=lambda w:svm_loss_naive(w,X_dev,y_dev,1e2)[0]#lambda是一个匿名函数
grad_numerical=grad_check_sparse(f,W,grad)

#执行vectorized version
from cs231n.classifiers.linear_svm import svm_loss_vectorized
tic=time.time()
loss_vec,_=svm_loss_vectorized(W,X_dev,y_dev,0.00001)
toc=time.time()
print("Vectorized loss: %f, time lasted: %fs" %(loss_vec,toc-tic))
print("loss difference: %f" %(loss-loss_vec))

# Stochastic Gradient descent随机梯度下降！！！
from cs231n.classifiers import LinearSVM
svm=LinearSVM()
tic=time.time()
loss_hist=svm.train(X_train,y_train,learning_rate=1e-7,reg=5e4,num_iters=1500)
toc=time.time()
print("That tooks: %fs" %(toc-tic))

#plot learning curve: plot loss as a function of iteration number
'''plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()'''

#预测结果，并输出准确率
y_train_pred=svm.predict(X_train)
print("Training accuracy: %f" %(np.mean(y_train_pred==y_train)))
y_val_pred=svm.predict(X_val)
print("validation accuracy: %f"%(np.mean(y_val==y_val_pred)))

#用validation set 调整超参数:正则化强度
learning_rates=[1e-7,5e-5]
regularizaion_strengths=[5e4,1e5]
results={}
best_val=-1
best_svm=None # The LinearSVM object that achieved the highest validation rate
for rate in learning_rates:
    for reg in regularizaion_strengths:
        svm_new=LinearSVM()
        loss=svm_new.train(X_train,y_train,rate,reg,1500)
        y_pred_val=svm_new.predict(X_val)
        y_pred_train=svm_new.predict(X_train)
        train_accuracy=np.mean(y_pred_train==y_train)
        val_accuracy=np.mean(y_pred_val==y_val)
        results[rate,reg]=(train_accuracy,val_accuracy)

        if val_accuracy>best_val:
            best_val=val_accuracy
            best_svm=svm_new

for lr,reg in sorted(results):
    train_accuracy,val_accuracy=results[(lr,reg)]
    print("lr %e reg %e \n train accuracy:%f val accuracy: %f"
          %(lr,reg,train_accuracy,val_accuracy))
print("best validation accuracy achieved during cross validation: %f" %best_val)


#最终在测试集上测试
y_test_pred=best_svm.predict(X_test)
test_accuracy=np.mean(y_test_pred==y_test)
print("SVM on raw pixels final test set accuracy: %f" %test_accuracy)