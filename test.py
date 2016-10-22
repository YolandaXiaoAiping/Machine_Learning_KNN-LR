from utils import *
import numpy as np

train_inputs, train_targets = load_train()
N,M = train_inputs.shape
weights = np.zeros(M+1)
weights_part = weights[0:M]
z_val = np.dot(train_inputs,weights_part)
z_val = z_val + weights[M:]
z_val = z_val.reshape(-1,1)
r1 = sigmoid(z_val).reshape(-1,1)

cross = -1*train_targets*np.log(r1)-(1-train_targets)*np.log(1-r1)
ce = cross.sum()
frac = ((r1>=0.5).astype(np.int)==train_targets).sum()/N

f = (-np.log(r1) + (1-train_targets)*z_val).sum()
wj_df = np.sum(((1-train_targets)*train_inputs -train_inputs*(1-r1)).T,axis=1).reshape(-1,1)
w0_df = np.sum((r1-train_targets).T,axis = 1).reshape(-1,1)
w=np.append(wj_df,w0_df,axis = 0)

print(wj_df.shape)
print(w0_df.shape)
#print(z_val.shape)
print(w.shape)
#print(train_inputs.shape)
#print(w0_df.shape)
#weight_decay = 0.01
#w2 = w + weight_decay*weights.reshape(-1,1)

#print(w2)
#print(w2.shape)
#print r1
#print r1.shape


