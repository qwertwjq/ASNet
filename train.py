import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from time import time
import os
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
BATCH_SIZE = 50

def get_data():
    raw_eeg = np.load('../data/train_input.npy')
    clean_eeg = np.load('../data/train_output.npy')
    
    train_input = np.concatenate((raw_eeg[:20000], raw_eeg[30000:50000]), axis=0)
    verify_input = np.concatenate((raw_eeg[20000:25000], raw_eeg[50000:55000]), axis=0)
    test_input = np.concatenate((raw_eeg[25000:30000], raw_eeg[55000:]), axis=0)

    train_output = np.concatenate((clean_eeg[:20000], clean_eeg[30000:50000]), axis=0)
    verify_output = np.concatenate((clean_eeg[20000:25000], clean_eeg[50000:55000]), axis=0)
    test_output = np.concatenate((clean_eeg[25000:30000], clean_eeg[55000:]), axis=0)


    
    train_input = torch.from_numpy(train_input)
    train_output = torch.from_numpy(train_output)
    
    train_torch_dataset = Data.TensorDataset(train_input, train_output)
    
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    verify_input = torch.from_numpy(verify_input)
    verify_output = torch.from_numpy(verify_output)
    
    verify_torch_dataset = Data.TensorDataset(verify_input, verify_output)
    
    verify_loader = Data.DataLoader(
        dataset=verify_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_input = torch.from_numpy(test_input)
    test_output = torch.from_numpy(test_output)
    
    test_torch_dataset = Data.TensorDataset(test_input, test_output)
    
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader,verify_loader,test_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    step_num=0
    loss_epoh=0
    for batch_idx, (train_input,train_output) in enumerate(train_loader):
        step_num +=1  
        train_input=train_input.float().to(device)
        train_output=train_output.float().to(device)
        optimizer.zero_grad()
        output = model(train_input)
        loss = loss_f(output, train_output)
        loss_epoh+=loss.item()
        loss.backward()
        optimizer.step()
         
    print(loss_epoh/step_num)
    return loss_epoh/step_num
            
def verify(model, device, verify_loader, optimizer, epoch):
    model.eval()
    step_num=0
    loss_epoh=0
    for batch_idx, (verify_input, verify_output) in enumerate(verify_loader):        
        step_num +=1
        verify_input=verify_input.float().to(device)
        verify_output=verify_output.float().to(device)
        output = model(verify_output)
        loss = loss_f(output, verify_output)
        loss_epoh+=loss.item()

    print(loss_epoh/step_num)
    return loss_epoh/step_num


train_loader,verify_loader,test_loader = get_data()
from ASNet import ASNet
model = ASNet()
model_name = 'ASNet'
learning_rate = 5e-4
loss_f = nn.MSELoss(reduction='mean')
print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
begin_time = time()

for epoch in range(150):
    train(model, device, train_loader, optimizer, epoch)
    verify(model, device, verify_loader, optimizer, epoch)
    print('save model')
    torch.save(model.state_dict(),  model_name + '.pkl')
    training_time = time() - begin_time
    minute = int(training_time // 60)
    second = int(training_time % 60)
    print(f'{minute}:{second}')
    print('epoch')
    print(epoch)
    print('finish')
