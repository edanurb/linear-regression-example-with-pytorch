import numpy as np 
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from  torch.utils.data import random_split
data= pd.read_csv("Marketing_Data.csv")
data_clean = np.array(data.dropna() ,dtype='float32')
x= data_clean[:,:3]
y= data_clean [ : , 3:]

x= torch.from_numpy(x)
y=torch.from_numpy(y)


dataset= TensorDataset(x,y)
train,test = random_split(dataset, [121,50])
batch_size=100
data_ld= DataLoader(train,batch_size )



model= nn.Linear(3,1)

epoch_size=10000
optimizer= torch.optim.SGD(model.parameters(),lr=1e-6)

for  epoch in range(epoch_size):
    
    for xd,yd in data_ld:
        #generate preds
        preds = model(xd)
        
        #loss
        loss=F.mse_loss(preds,yd)
        
        #graident decesent
        loss.backward()
        
        #optimezer
        optimizer.step()
        
        optimizer.zero_grad()
        
    if(epoch % 100 ==0):
        print("loss:  " + str(loss))
        
print("---------------- \n results")
result=  model(test[:][0]).detach().numpy()
real= test[:][1].detach().numpy()
print( list(model.parameters()))
# #
#  results
# [Parameter containing:
# tensor([[0.0496, 0.2427, 0.0153]], requires_grad=True), Parameter containing:
# tensor([0.2191], requires_grad=True)]


        