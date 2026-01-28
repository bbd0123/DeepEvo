import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os
from torch.nn.parallel import DataParallel
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore', category=FutureWarning)

## Function Definition
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)  
    predicted = predicted.cpu()
    _, predicted_true = torch.max(labels, 1)  
    predicted_true = predicted_true.cpu()
    correct = (predicted == predicted_true).sum().item()  
    accuracy = correct / predicted_true.size(0)  
    return accuracy

def data_augment_flip(tensor, tensor_labels):

    indices = torch.randperm(tensor.size(0))[:tensor.size(0) // 2]  
    selected_tensor = tensor[indices]
    selected_tensor_swapped = selected_tensor[:, [1, 0], :, :]  
    tensor[indices] = selected_tensor_swapped  

    for idx in indices:
        label = tensor_labels[idx]
        if torch.equal(label, torch.tensor([0,1,0])):
            tensor_labels[idx] = torch.tensor([0,0,1])
        elif torch.equal(label, torch.tensor([0,0,1])):
            tensor_labels[idx] = torch.tensor([0,1,0])

    return tensor, tensor_labels

def data_augment_maskPromoter(tensor, tensor_labels):
    
    promoter_start = int(tensor.shape[3]/2 - 20)
    promoter_end = int(tensor.shape[3]/2 + 20)
    
    indices = torch.randperm(tensor.size(0))[:tensor.size(0) // 2]  # 随机选择一半样本
    selected_tensor = tensor[indices]
    mean = 1
    std_dev = 0.3
    noise = np.random.normal(mean, std_dev, size = selected_tensor[:,:,:,promoter_start:promoter_end].shape).astype(np.float32)
    selected_tensor[:,:,:,promoter_start:promoter_end] = torch.tensor(noise)
    tensor[indices] = selected_tensor 

    return tensor, tensor_labels

def data_augment_maskEnhancer(tensor, tensor_labels):
    
    promoter_start = int(tensor.shape[3]/2 - 20)
    promoter_end = int(tensor.shape[3]/2 + 20)
    
    range_1 = range(0, promoter_start-39)
    range_2 = range(promoter_end, tensor.shape[3]-39)
    chosen_number = random.choice(list(range_1) + list(range_2))

    indices = torch.randperm(tensor.size(0))[:tensor.size(0) // 2]  
    selected_tensor = tensor[indices]
    mean = 1
    std_dev = 0.3
    noise = np.random.normal(mean, std_dev, size = selected_tensor[:,:,:,chosen_number:chosen_number+40].shape).astype(np.float32)
    selected_tensor[:,:,:,chosen_number:chosen_number+40] = torch.tensor(noise)
    tensor[indices] = selected_tensor  

    return tensor, tensor_labels

def data_augment_maskEpi(tensor, tensor_labels):

    chosen_number = random.choice([0,1,2,3,4])
    if chosen_number == 0:
        return tensor, tensor_labels
    else:
        chosen_epi = random.sample(range(0,5),chosen_number)
        mean = 1
        std_dev = 0.3
        noise = np.random.normal(mean, std_dev, size = tensor[:,:,chosen_epi,:].shape).astype(np.float32)
        tensor[:,:,chosen_epi,:] = torch.tensor(noise)
        return tensor, tensor_labels

def data_augment_addnoise(tensor, tensor_labels):
    mean = 0
    std_dev = random.choice([0,0.1,0.2,0.3,0.4,0.5])
    noise = np.random.normal(mean, std_dev, size = tensor[:,0,:,:].shape).astype(np.float32)
    
    original_shape = noise.shape
    flat_noise = noise.flatten()
    num_elements = flat_noise.size
    half_size = num_elements // 2
    indices = np.random.choice(num_elements, half_size, replace=False)
    flat_noise[indices] = 0
    noise = flat_noise.reshape(original_shape)
    
    tensor[:,0,:,:] = torch.tensor(noise) + tensor[:,0,:,:]
    
    
    std_dev = random.choice([0,0.1,0.2,0.3,0.4,0.5])
    noise = np.random.normal(mean, std_dev, size = tensor[:,1,:,:].shape).astype(np.float32)
    
    original_shape = noise.shape
    flat_noise = noise.flatten()
    num_elements = flat_noise.size
    half_size = num_elements // 2
    indices = np.random.choice(num_elements, half_size, replace=False)
    flat_noise[indices] = 0
    noise = flat_noise.reshape(original_shape)    

    tensor[:,1,:,:] = torch.tensor(noise) + tensor[:,1,:,:]
    return tensor, tensor_labels
    
def data_augment_ValTest(tensor, tensor_labels):

    length = tensor.shape[0]
    tensor_swapped = tensor[:, [1, 0], :, :]  
    tensor_labels_swapped = tensor_labels.clone()

    for idx in range(length):
        label = tensor_labels[idx]
        if torch.equal(label, torch.tensor([0,1,0])):
            tensor_labels_swapped[idx] = torch.tensor([0,0,1])
        elif torch.equal(label, torch.tensor([0,0,1])):
            tensor_labels_swapped[idx] = torch.tensor([0,1,0])

    return torch.cat((tensor,tensor_swapped), dim=0), torch.cat((tensor_labels,tensor_labels_swapped), dim=0)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def train_model(model, criterion, criterion_val, optimizer, epochs, batch_size,
                sequences_train, labels_train, sequences_val, labels_val, suffix, device, if_data_augment=True):

    early_stopping = EarlyStopping(patience=10)
    dataset = TensorDataset(sequences_train, labels_train)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_train = 0
        train_acc_batch = []
        all_preds = []
        all_labels = []
        for batch_sequences, batch_labels in dataloader:
            if if_data_augment:
                batch_sequences, batch_labels = data_augment_flip(batch_sequences, batch_labels)
                batch_sequences, batch_labels = data_augment_maskPromoter(batch_sequences, batch_labels)
                batch_sequences, batch_labels = data_augment_maskEnhancer(batch_sequences, batch_labels)
                batch_sequences, batch_labels = data_augment_maskEpi(batch_sequences, batch_labels)
                batch_sequences, batch_labels = data_augment_addnoise(batch_sequences, batch_labels)
                
                
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_sequences[:,0],batch_sequences[:,1])
            accu_train = calculate_accuracy(outputs, batch_labels)
            train_acc_batch.append(accu_train)
            
            integer_labels = torch.argmax(batch_labels, dim=1)
            loss = criterion(outputs, integer_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        average_loss = total_loss / len(dataloader)
        train_loss.append(average_loss)
        average_train_acc = np.mean(train_acc_batch)
        print(f'Epoch {epoch + 1}/{epochs}, Train - Average Loss: {average_loss}, ACC: {average_train_acc}')
        
        val_loss_batch = []
        val_acc_batch = []
        
        val_num = sequences_val.shape[0]
        for i in range(val_num):
            with torch.no_grad():
                outputs_val = model(sequences_val[i:(i+1),0],sequences_val[i:(i+1),1])
                integer_labels = torch.argmax(labels_val[i:i+1], dim=1)
                loss_val = criterion_val(outputs_val, integer_labels)
                accu_val = calculate_accuracy(outputs_val, labels_val[i:i+1])
            val_loss_batch.append(loss_val.item())
            val_acc_batch.append(accu_val)
        
        average_val_loss = np.mean(val_loss_batch)
        average_val_acc = np.mean(val_acc_batch)
        early_stopping(average_val_loss, epoch+1)
        if early_stopping.early_stop:
            print("Early stopping")
            print(early_stopping.best_epoch, early_stopping.best_loss)
            break
        
        val_loss.append(average_val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Validation - Average Loss: {average_val_loss}, ACC: {average_val_acc}')
        
        torch.save(model, './model_'+suffix+'_epoch'+str(epoch+1)+'.pth')
        
    train_loss = [float(i) for i in train_loss]
    val_loss = [float(i) for i in val_loss]
    return train_loss, val_loss, early_stopping.best_epoch, early_stopping.best_loss

def cal_weight(label_tensor_train):
    noDEG_num = 0
    up_num = 0
    down_num = 0
    for idx in range(label_tensor_train.shape[0]):
        label = label_tensor_train[idx]
        if torch.equal(label, torch.tensor([1,0,0])):
            noDEG_num = noDEG_num + 1
        elif torch.equal(label, torch.tensor([0,1,0])):
            up_num = up_num + 1
        else:
            down_num = down_num + 1
    print(noDEG_num,up_num,down_num)
    sum_num = (noDEG_num + up_num + down_num)
    print(1-(noDEG_num /sum_num), 1-(up_num/sum_num), 1-(down_num/sum_num))
    print((sum_num/noDEG_num), (sum_num/up_num), (sum_num/down_num))
    return (sum_num/noDEG_num), (sum_num/up_num), (sum_num/down_num)

def model_performance(model, sequences_val, labels_val):
    number = sequences_val.shape[0]
    label_true = []
    label_predict = []
    for i in range(number):
        with torch.no_grad():
            outputs_val = model(sequences_val[i:(i+1),0],sequences_val[i:(i+1),1])
            _,pre = torch.max(outputs_val, 1)
            pre = pre.cpu().numpy()[0]
            label_predict.append(pre)
            
            if torch.equal(labels_val[i], torch.tensor([1,0,0])):
                label_true.append(0)
            elif torch.equal(labels_val[i], torch.tensor([0,1,0])):
                label_true.append(1)
            else:
                label_true.append(2)
    measure_result = classification_report(label_true, label_predict,digits=6,output_dict=True)
    
    
    return label_true,label_predict,measure_result

def model_performance_run(suffix,sequence_tensor_test,label_tensor_test,epoch_num=20):
    result_dict = {}
    for epoch in range(epoch_num):
        result_dict_tmp = {}
        model1_best = torch.load('/home/user/data3/qij/project/inversion/epigenomics_data/feature_extract_v2/Model/model_'+suffix+'_epoch'+str(epoch+1)+'.pth')
        model1_best = model1_best.eval()
        label_true,label_predict,acc,measure_result1 = model_performance(model1_best, sequence_tensor_test, label_tensor_test)
        result_dict_tmp['label_true'] = label_true
        result_dict_tmp['label_predict'] = label_predict
        result_dict_tmp['Acc'] = acc
        result_dict_tmp['measure_result'] = measure_result1
        result_dict[epoch] = result_dict_tmp
    return result_dict

def model_performance_run_forOne(suffix,sequence_tensor_test,label_tensor_test,epoch):
    model1_best = torch.load('./model_'+suffix+'_epoch'+str(epoch)+'.pth')
    model1_best = model1_best.eval()
    label_true,label_predict,measure_result1 = model_performance(model1_best, sequence_tensor_test, label_tensor_test)
    return measure_result1

def train_model_run(path, sequence_tensor_train, label_tensor_train, sequence_tensor_val, label_tensor_val, suffix, length, epoch_num=20, weight=False, if_data_augment=True, lr=1e-4, weight_decay=1e-4):
    device=torch.device('cuda:0')
    model = torch.load(path)
    model = model.module.to(device)
    model = DataParallel(model, device_ids=[0,1])
    
    weight1,weight2,weight3 = cal_weight(label_tensor_train)
    class_weights = torch.tensor([weight1,weight2,weight3]).to(device)
    # 损失函数和优化器
    if weight:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    weight1,weight2,weight3 = cal_weight(label_tensor_val)
    class_weights_val = torch.tensor([weight1,weight2,weight3]).to(device)
    criterion_val = nn.CrossEntropyLoss(weight=class_weights_val)
    optimizer = optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    label_tensor_val = label_tensor_val.to(device)
    train1_loss, val1_loss, best_epoch, best_loss = \
    train_model(model, criterion, criterion_val, optimizer, epoch_num, 64,
                    sequence_tensor_train, label_tensor_train, 
                    sequence_tensor_val, label_tensor_val, suffix, device, if_data_augment=if_data_augment)
    return train1_loss, val1_loss, best_epoch, best_loss

## Model
class Stransformer(nn.Module):
    def __init__(self, size):
        super(Stransformer, self).__init__()
        
        self.conv1d_shared1 = nn.Conv1d(5, 32, kernel_size=1, stride=1)
        self.conv1d_shared2 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        self.conv1d_shared3 = nn.Conv1d(64, 64, kernel_size=1, stride=1)
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.pool2 = nn.AvgPool1d(kernel_size=8, stride=8)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * size, 32)
        self.fc2 = nn.Linear(64, 3)

    def forward_once(self, x):
        x = self.conv1d_shared1(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.conv1d_shared2(x)
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.conv1d_shared3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), dim=1)
        output = self.fc2(output)
        #output = torch.softmax(output,dim=1)
        return output
    
## data
def balances_data(epi_tensor_train, label_tensor_train):
    class_1_idx = (label_tensor_train[:, 0] == 1).nonzero(as_tuple=True)[0]
    class_2_idx = (label_tensor_train[:, 1] == 1).nonzero(as_tuple=True)[0]
    class_3_idx = (label_tensor_train[:, 2] == 1).nonzero(as_tuple=True)[0]
    
    # 找到最少类别的数量
    min_count = min(len(class_1_idx), len(class_2_idx), len(class_3_idx))
    print(min_count)
    
    # 对每个类别进行随机采样，使得它们的数量一致
    class_1_idx_sampled = class_1_idx[torch.randperm(len(class_1_idx))[:min_count]]
    class_2_idx_sampled = class_2_idx[torch.randperm(len(class_2_idx))[:min_count]]
    class_3_idx_sampled = class_3_idx[torch.randperm(len(class_3_idx))[:min_count]]
    
    # 将采样后的索引合并
    sampled_indices = torch.cat((class_1_idx_sampled, class_2_idx_sampled, class_3_idx_sampled))
    # 使用这些索引从原始数据和标签中提取样本
    balanced_epi_tensor_train = epi_tensor_train[sampled_indices]
    balanced_label_tensor_train = label_tensor_train[sampled_indices]
    
    return balanced_epi_tensor_train, balanced_label_tensor_train

with open('./preepi_tensor_dict.json', 'rb') as fp:
    preepi_tensor_dict = pickle.load(fp)
with open('./label_tensor_dict.json', 'rb') as fp:
    label_tensor_dict = pickle.load(fp)
    
preepi_tensor_train = torch.cat([preepi_tensor_dict[key]['Train'] for key in preepi_tensor_dict])
label_tensor_train = torch.cat([label_tensor_dict[key]['Train'] for key in label_tensor_dict])

preepi_tensor_val = torch.cat([preepi_tensor_dict[key]['Val'] for key in preepi_tensor_dict])
label_tensor_val = torch.cat([label_tensor_dict[key]['Val'] for key in label_tensor_dict])

preepi_tensor_test = torch.cat([preepi_tensor_dict[key]['Test'] for key in preepi_tensor_dict])
label_tensor_test = torch.cat([label_tensor_dict[key]['Test'] for key in label_tensor_dict])

preepi_tensor_val_augment, label_tensor_val_augment = data_augment_ValTest(preepi_tensor_val, label_tensor_val)
preepi_tensor_test_augment, label_tensor_test_augment = data_augment_ValTest(preepi_tensor_test, label_tensor_test)
preepi_tensor_train_balanced, label_tensor_train_balanced = balances_data(preepi_tensor_train, label_tensor_train)

Sequence_tensor_train = preepi_tensor_train_balanced
Sequence_tensor_val = preepi_tensor_val_augment
Label_tensor_train = label_tensor_train_balanced
Label_tensor_val = label_tensor_val_augment

## Model training
suffix = 'LengthTest1152'
path = './Model/model_'+suffix+'.pth'

weight_decay=1e-3
epoch_num = 50000
suffix = 'Model_finetuning
length_res['train_loss'], length_res['val_loss'], length_res['best_epoch'], length_res['best_loss'] = \
train_model_run(path, Sequence_tensor_train, Label_tensor_train, Sequence_tensor_val, Label_tensor_val,\
                suffix, length, epoch_num, weight=False, lr=5e-5, weight_decay=weight_decay)
