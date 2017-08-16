"""
In this exercise we will implement fully-connected networks using a more
modular approach. For each layer we will implement a forward and a
backward function. The forward function will receive inputs, weights,
and other parameters and will return both an output and a cache object
storing data needed for the backward pass, like this:
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import  *
from cs231n.data_utils import  load_CIFAR10
from cs231n.gradient_check import eval_numerical_gradient,eval_numerical_gradient_array
from cs231n.solver import Solver

def rel_error(x,y):
    #返回相对误差
    return np.max(np.abs(x-y)/np.maximum(1e-8,np.abs(x)+np.abs(y)))

#读取数据
data =load_CIFAR10('E:\\Python\\deep learning CS231n\\assignment2\\cs231n\\datasets')
for k, v in data.items():
  print('%s: ' % k, v.shape)

#完成layers.py
#检验
num_inputs=2
input_shape=(4,5,6)
ouput_dim=3
input_size=num_inputs*np.prod(input_shape)
weight_size=np.prod(input_shape)*ouput_dim
x=np.linspace(-0.1,0.5,num=input_size).reshape(num_inputs,*input_shape)
w=np.linspace(-0.2,0.3,num=weight_size).reshape(np.prod(input_shape),ouput_dim)
b=np.linspace(-0.3,0.1,num=ouput_dim)

out,_=affine_forward(x,w,b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])
print('与正确的out差别是：',rel_error(out,correct_out))


#现在完成反向传播 affine_backward
#验证
x=np.random.rand(10,2,3)
w=np.random.randn(6,5)
b=np.random.randn(5)
dout=np.random.randn(10,5)
dx_num=eval_numerical_gradient_array(lambda x: affine_forward(x,w,b)[0],x,dout)
dw_num=eval_numerical_gradient_array(lambda w:affine_forward(x,w,b)[0],w,dout)
db_num=eval_numerical_gradient_array(lambda b:affine_forward(x,w,b)[0],b,dout)
cache=(x,w,b)
dx,dw,db=affine_backward(dout,cache)
print('Testing 反向传播：')
print('dx error:',rel_error(dx,dx_num))
print ('dw error: ', rel_error(dw_num, dw))
print ('db error: ', rel_error(db_num, db))

#完成 relu layer 前向传播,反向传播
#将affine layer 和 relu layer连在一起

'''from cs231n.layer_utils import affine_relu_forward, affine_relu_backward

x = np.random.randn(2, 3, 4)
w = np.random.randn(12, 10)
b = np.random.randn(10)
dout = np.random.randn(2, 10)

out, cache = affine_relu_forward(x, w, b)
dx, dw, db = affine_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

print ('Testing affine_relu_forward:')
print ('dx error: ', rel_error(dx_num, dx))
print ('dw error: ', rel_error(dw_num, dw))
print ('db error: ', rel_error(db_num, db))

#loss layer: Softmax and SVM
x=0.001* np.random.randn(50,10) #前向传播后的值
y=np.random.randint(10,size=50) #在0~10之间随机选一个整数，共有50个
dx_num=eval_numerical_gradient(lambda x: svm_loss(x,y)[0],x,verbose=False)
loss,dx=svm_loss(x,y)
print('SVM difference:',rel_error(dx_num,dx))'''

#接下来完成fc_net,用上述部件完成两层网络
#测试
'''N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-2
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

print( 'Testing initialization ... ')
W1_std = abs(model.params['W1'].std() - std)
b1 = model.params['b1']
W2_std = abs(model.params['W2'].std() - std)
b2 = model.params['b2']
assert W1_std < std / 10, 'First layer weights do not seem right'
assert np.all(b1 == 0), 'First layer biases do not seem right'
assert W2_std < std / 10, 'Second layer weights do not seem right'
assert np.all(b2 == 0), 'Second layer biases do not seem right'

print ('Testing test-time forward pass ... ')
model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff < 1e-6, 'Problem with test-time forward pass'

print ('Testing training loss (no regularization)')
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

for reg in [0.0, 0.1, 0.7]:
  print ('Running numeric gradient check with reg = ', reg)
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))'''

#用Solver.py做训练
'''model=TwoLayerNet(reg=0.4)
solver=Solver(model,data,update_rule='sgd',optim_config={
    'learning_rate':1e-3,},lr_decay=0.95,num_epochs=10,batch_size=100,print_every=100)
solver.train()'''

#fullyconnected layer
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for reg in [0, 3.14]:
  print ('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)

  loss, grads = model.loss(X, y)
  print ('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#用动量+SGD更新参数
'''from cs231n.optim import sgd_momentum

N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

config = {'learning_rate': 1e-3, 'velocity': v}
next_w, _ = sgd_momentum(w, dw, config=config)

expected_next_w = np.asarray([
  [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
  [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
  [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
  [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
expected_velocity = np.asarray([
  [ 0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
  [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
  [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
  [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])

print ('next_w error: ', rel_error(next_w, expected_next_w))
print ('velocity error: ', rel_error(expected_velocity, config['velocity']))'''

#用SGD+momentum做训练更快收敛

'''num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
solvers={}

for update_rule in ['sgd','sgd_momentum']:
    print('running with: ',update_rule)
    model=FullyConnectedNet([100,100,100,100,100],weight_scale=5e-2,reg=0.05);
    solver=Solver(model,small_data,num_epochs=5,batch_size=100,update_rule=update_rule,optim_config={'learning_rate':1e-2,},verbose=True)
    solvers[update_rule]=solver
    solver.train()'''

#test RMSProp
'''from cs231n.optim import rmsprop
N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
cache = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

config = {'learning_rate': 1e-2, 'cache': cache}
next_w,_ = rmsprop(w, dw, config=config)

expected_next_w = np.asarray([
  [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
  [-0.132737,   -0.08078555, -0.02881884,  0.02316247,  0.07515774],
  [ 0.12716641,  0.17918792,  0.23122175,  0.28326742,  0.33532447],
  [ 0.38739248,  0.43947102,  0.49155973,  0.54365823,  0.59576619]])
expected_cache = np.asarray([
  [ 0.5976,      0.6126277,   0.6277108,   0.64284931,  0.65804321],
  [ 0.67329252,  0.68859723,  0.70395734,  0.71937285,  0.73484377],
  [ 0.75037008,  0.7659518,   0.78158892,  0.79728144,  0.81302936],
  [ 0.82883269,  0.84469141,  0.86060554,  0.87657507,  0.8926    ]])

print ('next_w error: ', rel_error(expected_next_w, next_w))
print ('cache error: ', rel_error(expected_cache, config['cache']))'''

#test Adam
'''from cs231n.optim import adam

N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)

config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
next_w, _ = adam(w, dw, config=config)

expected_next_w = np.asarray([
  [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
  [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929],
  [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
  [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
expected_v = np.asarray([
  [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
  [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
  [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
  [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966,   ]])
expected_m = np.asarray([
  [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
  [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
  [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
  [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])

print ('next_w error: ', rel_error(expected_next_w, next_w))
print ('v error: ', rel_error(expected_v, config['v']))
print ('m error: ', rel_error(expected_m, config['m']))'''


#检验batch normalization 前向传播
'''# Simulate the forward pass for a two-layer network
N, D1, D2, D3 = 200, 50, 60, 3
X = np.random.randn(N, D1)
W1 = np.random.randn(D1, D2)
W2 = np.random.randn(D2, D3)
a = np.maximum(0, X.dot(W1)).dot(W2)

print ('Before batch normalization:')
print ('  means: ', a.mean(axis=0))
print ('  stds: ', a.std(axis=0))

# Means should be close to zero and stds close to one
print ('After batch normalization (gamma=1, beta=0)')
a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
print ('  mean: ', a_norm.mean(axis=0))
print ('  std: ', a_norm.std(axis=0))

# Now means should be close to beta and stds close to gamma
gamma = np.asarray([1.0, 2.0, 3.0])
beta = np.asarray([11.0, 12.0, 13.0])
a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
print( 'After batch normalization (nontrivial gamma, beta)')
print ('  means: ', a_norm.mean(axis=0))
print ('  stds: ', a_norm.std(axis=0))'''
#检验BN反向传播
'''N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for reg in [0, 3.14]:
  print ('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64,
                            use_batchnorm=True)

  loss, grads = model.loss(X, y)
  print ('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
  if reg == 0: pass'''


#检验dropout正确性
'''N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for dropout in [0, 0.25, 0.5]:
  print ('Running check with dropout = ', dropout)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            weight_scale=5e-2, dtype=np.float64,
                            dropout=dropout, seed=123)

  loss, grads = model.loss(X, y)
  print ('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))'''
#检验Dropout的效果
'''num_train=500
small_data={
  'X_train':data['X_train'][:num_train],
  'y_train':data['y_train'][:num_train],
  'X_val':data['X_val'],
  'y_val':data['y_val'],
}
solvers={}
dropout_choices=[0,0.75]
for dropout in dropout_choices:
  model=FullyConnectedNet([500],dropout=dropout)
  print('dropout: %f' %dropout)
  solver=Solver(model,small_data,num_epochs=25,batch_size=100,update_rule='adam',optim_config={'learning_rate':5e-4},verbose=True,print_every=100)
  solver.train()
  solvers[dropout]=solver
  '''


#train a good model

best_model=None
X_val=data['X_val']
y_val=data['y_val']
X_test=data['X_test']
y_test=data['y_test']
lr=1e-04
model=FullyConnectedNet([100,100,100,100],weight_scale=2.4618e-02,dtype=np.float64,use_batchnorm=True,dropout=0.5,reg=1e-2)
solver=Solver(model,data,print_every=100,num_epochs=10,batch_size=25,update_rule='adam',optim_config={'learning_rate':lr,},lr_decay=1.0,verbose=True)
solver.train()


plt.subplot(2, 1, 1)
plt.plot(solver.loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, label='train')
plt.plot(solver.val_acc_history, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

best_model = model