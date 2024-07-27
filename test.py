import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from time import time
import os
import zhibiao
import matplotlib.pyplot as plt
from ASNet import ASNet

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




def test(model, device, test_loader,num,num_x,input_z ,output_z,pre_z):
    model.eval()
    step_num=0
    loss_epoh=0
    for batch_idx, (test_input, indicator, test_output) in enumerate(test_loader):
        indicator = indicator.float().to(device)
        test_input=test_input.float().to(device)
        test_output=test_output.float().to(device)
        output,en_x,fl_x_s,decoder_x= model(test_input)
        output = output.detach().cpu()
        test_output = test_output.detach().cpu()
        test_input = test_input.detach().cpu()      
        
        input_z[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE] = test_input
        
        output_z[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE] = test_output
        
        pre_z[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE] = output
        step_num +=1


train_loader,verify_loader,test_loader = get_data()
model = ASNet()
model_name = 'ASNet'

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

begin_time = time()
if os.path.exists( model_name + '.pkl'):
    print('load model')
    model.load_state_dict(torch.load( model_name + '.pkl'))
num = 10000
num_x = num//10
test_input_z = torch.ones(num,512)
test_output_z = torch.ones(num,512)  
pre_z = torch.ones(num,512)
test(model, device, test_loader,num,num/10,test_input_z ,test_output_z ,pre_z )


i=100
x = np.linspace(0, 2, 512)
l0, = plt.plot(x, test_input_z[i])
l1, = plt.plot(x, test_output_z[i])
l2, = plt.plot(x, pre_z[i])
plt.legend([l0, l1, l2], ['Contamineted EEG', 'Pure EEG', 'Corrected EEG'], loc='upper right')
plt.xlabel('Time (s)')  # 设置x轴标签
plt.ylabel('Amplitude(mV)')  # 设置y轴标签


