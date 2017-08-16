import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

def rel_error(x,y):
    #returns relative error
    return np.max(np.abs(x-y) / (np.maximum(1e-8,np.abs(x)+np.abs(y))))
'''
#创建一个小网络来用作检查
#Note that we set the random seed for repeatable experiments

input_size=4
hidden_size=10
num_classes=3
num_inputs=5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size,hidden_size,num_classes,std=1e-1)
def init_toy_data():
    np.random.seed(1)
    X=10*np.random.randn(num_inputs,input_size)#随机生成一个5*4的矩阵
    y=np.array([0,1,2,2,1])
    return X,y
net=init_toy_model()
X,y=init_toy_data()

#前向传播：算出scores/loss
scores=net.loss(X)#输入参数中没有y,则得到的是每个样本对应每个类别的分数
print('Your scores:')
print (scores)
print("correct scores:")
correct_scores=np.array([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print (correct_scores)
print('Different between your scores and correct scores:')
print(np.sum(np.abs(scores-correct_scores)))

#算loss
loss,grad=net.loss(X,y,reg=0.1)
correct_loss=1.30378789133

print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss-correct_loss)))

#梯度检查
from cs231n.gradient_check import eval_numerical_gradient
for param_name in grad:
    f=lambda W: net.loss(X,y,reg=0.1)[0]
    param_grad_num=eval_numerical_gradient(f,net.params[param_name],verbose=False)
    print('%s max relative error: %e'%(param_name,rel_error(param_grad_num,grad[param_name])))

#train the network
#用随机梯度下降
net=init_toy_model()
stats=net.train(X,y,X,y,learning_rate=1e-1,reg=1e-5,num_iters=100,verbose=False)#返回的是个dict

print('Final training loss:',stats['loss_history'][-1])
#plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training loss history')
plt.show()
'''
#小网络检查完毕
#导入真实数据
from cs231n.data_utils import load_CIFAR10
def get_CIFAR10_data(num_training=9000,num_validation=1000,num_test=1000):
    cifar10_dir='cs231n//datasets'
    X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)

    #subsample
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)

input_size=32*32*3
hidden_size=50
num_classes=50
net=TwoLayerNet(input_size,hidden_size,num_classes)
#train the model
stats=net.train(X_train,y_train,X_val,y_val,num_iters=1000,batch_size=200,learning_rate=1e-4,learning_rate_decay=0.95,reg=0.5,verbose=True)

#predict
val_acc=(net.predict(X_val)==y_val).mean()
print('Validation accuracy: ',val_acc)

#得到0.278不是很好，所以我们要plot loss funtion and accuracies on the training and validation set during optimization

#plot the loss function and train/validation accuracies
plt.subplot(2,1,1)
plt.plot(stats['loss_history'])
plt.title('loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2,1,2)
plt.plot(stats['train_acc_history'],label='train')
plt.plot(stats['val_acc_history'],label='val')
plt.title('Claasification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.show()

#看第一层的weights
from cs231n.vis_utils import visualize_grid
def show_net_weights(net):
    W1=net.params['W1']
    W1=W1.reshape(32,32,3,-1).transpose(3,0,1,2)
    plt.imshow(visualize_grid(W1,padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
show_net_weights(net)

#调整超参数
best_net=None #store the best model into this
#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
learning_rate=[3e-4,1e-2,3e-1]
hidden_size=[30,50,100]
train_epoch=[30,50]
reg=[0.1,3,10]
best_val_acc=-1
best={}
for each_lr in learning_rate:
    for each_hid in hidden_size:
        for each_epo in train_epoch:
            for each_reg in reg:
                net=TwoLayerNet(input_size,each_hid,num_classes)
                stats=net.train(X_train,y_train,X_val,y_val,learning_rate=each_lr,reg=each_reg,num_epochs=each_epo)
                train_acc=stats['train_acc_history'][-1]
                val_acc=stats['val_acc_history'][-1]
                if val_acc>best_val_acc:
                    best_val_acc=val_acc
                    best_net=net
                    best['learning_rate']=each_lr
                    best['hidden_size']=each_hid
                    best['epoch']=each_epo
                    best['reg']=each_reg

for each_key in best:
    print('best net:\n '+each_key+':%e'%best[each_key])

print('best validation accuracy: %f' %best_val_acc)



#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################