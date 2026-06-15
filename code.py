import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# this transformation going to apply on all images
transform=transforms.Compose([
    #imge to tensor and normalize (-1,1)
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainset=CIFAR10(root="./data",train=True,download=True,transform=transform)
testnset=CIFAR10(root="./data",train=False,download=True,transform=transform)
train_loader=DataLoader(trainset,batch_size=64,shuffle=True)
test_loader=DataLoader(testnset,batch_size=64)
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv_layers=nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
            
        nn.Conv2d(32,64,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),   

        nn.Conv2d(64,128,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
        self.fc_layer=nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self,x):
        x=self.conv_layers(x)
        x=x.view(x.size(0),-1) # flattening
        x=self.fc_layer(x)
        return x


model=CNN()
criteria=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters())
epochs=20
for epoch in range(epochs):
    train_loss=0.0
    for image , labels in train_loader:
        optimizer.zero_grad()
        output=model.forward(image)
        loss=criteria(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    print(f"epoch={epoch+1} & loss {train_loss}")
  # evaluate
current=0
total=0
model.eval()
with torch.no_grad():
    for image,label in test_loader:
        output=model.forward(image)
        _,predict=torch.max(output,1)
        current+=(predict==label).sum().item()
        total+=label.size(0)
    print(f"accuracy {current/total *100}")

        

