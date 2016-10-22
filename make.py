import numpy as np
from run_knn import run_knn
from utils import *
from l2_distance import l2_distance
import matplotlib.pyplot as plt

train_d,train_l = load_train()
train_small_d,train_small_l = load_train_small()
valid_d,valid_l = load_valid()
test_d,test_l = load_test()

kNN = [1,3,5,7,9]
res_train = []
res_small_train = []

for k in kNN:
 e = (run_knn(k,train_d,train_l,valid_d)==valid_l)
 f = e.sum()*1.0/e.shape[0]
 res_train.append([k,f])

for k in kNN:
 g = (run_knn(k,train_small_d,train_small_l,valid_d)==valid_l)
 h = g.sum()*1.0/g.shape[0]
 res_small_train.append([k,h])

print(res_train)
print(res_small_train)
#plt.ylim(0.90,1)
plt.ylim(0.4,1)
plt.plot(*zip(*res_train),marker='o',color='r',ls='-')
plt.plot(*zip(*res_small_train),marker='o',color='r',ls='--')

plt.show()

test_k = [3,5,7]
res1 = []

for y in test_k:
	m = (run_knn(y,train_d,train_l,test_d) == test_l)
	n = m.sum()*1.0/m.shape[0]
	res1.append([y,n])

print(res1)
plt.ylim(0.9,1)
plt.plot(*zip(*res1),marker='o',color='r',ls='-')
plt.show()