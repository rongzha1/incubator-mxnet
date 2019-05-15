import mxnet as mx
import numpy as np
import os
import time

data = mx.nd.arange(0,64).reshape((2,8,2,2))
'''
data_np = data.asnumpy()
print(data_np)
mean_data = np.mean(data_np, axis=0)
print(mean_data)
var_data = np.var(data_np, axis=0)
print(var_data)
eps = 0
gamma = np.ones_like(mean_data)
beta = np.zeros_like(mean_data)
out = ((data_np - mean_data)/(np.sqrt(var_data)+eps) ) * gamma + beta
print (out)
'''

#data = mx.nd.array([[1, 2], [3, 6]])
#data = mx.nd.ones((2,8,2,2))
#print(data.asnumpy())
mean_data = mx.nd.mean(data, axis=0)
#print(mean_data.asnumpy())

a = data - mean_data
#var = 
sqr_a = mx.nd.square(a)
#print (sqr_a.asnumpy())
var_data = mx.nd.mean(sqr_a, axis=0)
#print(var_data.asnumpy())
eps = 1e-3
gamma = mx.nd.ones_like(mean_data)
beta = mx.nd.zeros_like(mean_data)
out = ((data - mean_data)/(mx.nd.sqrt(var_data)+eps) ) * gamma + beta
#print (out.asnumpy())

os.environ["MXNET_MKLDNN_ENABLED"] = "True"
bn_mkl = mx.nd.BatchNorm(data=data, gamma=gamma, beta=beta, moving_mean=mean_data, moving_var=var_data, eps=eps, use_global_stats=1)

os.environ["MXNET_MKLDNN_ENABLED"] = "False"
bn_native = mx.nd.BatchNorm(data=data, gamma=gamma, beta=beta, moving_mean=mean_data, moving_var=var_data, eps=eps, use_global_stats=1)

assert ((bn_native.asnumpy() == bn_mkl.asnumpy()).all()), "case fail"

