import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from plotnine import *
import random
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from collections import Counter


## Read data
def find_overlap(list1, list2, list3, list4, list5, cutoff):
    all_numbers = list1 + list2 + list3 + list4 + list5
    counter = Counter(all_numbers)
    result = [num for num, count in counter.items() if count >= cutoff]
    return result

def filter_idx(labels_train_homsap, labels_val_homsap, labels_test_homsap, percentile, cutoff):
    epi0_mean_list_tarin = []
    epi1_mean_list_tarin = []
    epi2_mean_list_tarin = []
    epi3_mean_list_tarin = []
    epi4_mean_list_tarin = []
    for num in range(labels_train_homsap.shape[0]):
        epi0_mean_list_tarin.append(float(labels_train_homsap[num,0].mean()))
        epi1_mean_list_tarin.append(float(labels_train_homsap[num,1].mean()))
        epi2_mean_list_tarin.append(float(labels_train_homsap[num,2].mean()))
        epi3_mean_list_tarin.append(float(labels_train_homsap[num,3].mean()))
        epi4_mean_list_tarin.append(float(labels_train_homsap[num,4].mean()))
    epi0_mean_list_val = []
    epi1_mean_list_val = []
    epi2_mean_list_val = []
    epi3_mean_list_val = []
    epi4_mean_list_val = []
    for num in range(labels_val_homsap.shape[0]):
        epi0_mean_list_val.append(float(labels_val_homsap[num,0].mean()))
        epi1_mean_list_val.append(float(labels_val_homsap[num,1].mean()))
        epi2_mean_list_val.append(float(labels_val_homsap[num,2].mean()))
        epi3_mean_list_val.append(float(labels_val_homsap[num,3].mean()))
        epi4_mean_list_val.append(float(labels_val_homsap[num,4].mean()))
    epi0_mean_list_test = []
    epi1_mean_list_test = []
    epi2_mean_list_test = []
    epi3_mean_list_test = []
    epi4_mean_list_test = []
    for num in range(labels_test_homsap.shape[0]):
        epi0_mean_list_test.append(float(labels_test_homsap[num,0].mean()))
        epi1_mean_list_test.append(float(labels_test_homsap[num,1].mean()))
        epi2_mean_list_test.append(float(labels_test_homsap[num,2].mean()))
        epi3_mean_list_test.append(float(labels_test_homsap[num,3].mean()))
        epi4_mean_list_test.append(float(labels_test_homsap[num,4].mean()))
        
    epi0_cutoff = np.percentile(epi0_mean_list_tarin+epi0_mean_list_val+epi0_mean_list_test,percentile)
    epi1_cutoff = np.percentile(epi1_mean_list_tarin+epi1_mean_list_val+epi1_mean_list_test,percentile)
    epi2_cutoff = np.percentile(epi2_mean_list_tarin+epi2_mean_list_val+epi2_mean_list_test,percentile)
    epi3_cutoff = np.percentile(epi3_mean_list_tarin+epi3_mean_list_val+epi3_mean_list_test,percentile)
    epi4_cutoff = np.percentile(epi4_mean_list_tarin+epi4_mean_list_val+epi4_mean_list_test,percentile)
    
    epi0_filter_tarin = list(np.where(np.array(epi0_mean_list_tarin) < epi0_cutoff)[0])
    epi1_filter_tarin = list(np.where(np.array(epi1_mean_list_tarin) < epi1_cutoff)[0])
    epi2_filter_tarin = list(np.where(np.array(epi2_mean_list_tarin) < epi2_cutoff)[0])
    epi3_filter_tarin = list(np.where(np.array(epi3_mean_list_tarin) < epi3_cutoff)[0])
    epi4_filter_tarin = list(np.where(np.array(epi4_mean_list_tarin) < epi4_cutoff)[0])

    epi0_filter_val = list(np.where(np.array(epi0_mean_list_val) < epi0_cutoff)[0])
    epi1_filter_val = list(np.where(np.array(epi1_mean_list_val) < epi1_cutoff)[0])
    epi2_filter_val = list(np.where(np.array(epi2_mean_list_val) < epi2_cutoff)[0])
    epi3_filter_val = list(np.where(np.array(epi3_mean_list_val) < epi3_cutoff)[0])
    epi4_filter_val = list(np.where(np.array(epi4_mean_list_val) < epi4_cutoff)[0])

    epi0_filter_test = list(np.where(np.array(epi0_mean_list_test) < epi0_cutoff)[0])
    epi1_filter_test = list(np.where(np.array(epi1_mean_list_test) < epi1_cutoff)[0])
    epi2_filter_test = list(np.where(np.array(epi2_mean_list_test) < epi2_cutoff)[0])
    epi3_filter_test = list(np.where(np.array(epi3_mean_list_test) < epi3_cutoff)[0])
    epi4_filter_test = list(np.where(np.array(epi4_mean_list_test) < epi4_cutoff)[0])
    
    train_filter_list = find_overlap(epi0_filter_tarin, epi1_filter_tarin, epi2_filter_tarin, epi3_filter_tarin, epi4_filter_tarin, cutoff)
    val_filter_list = find_overlap(epi0_filter_val, epi1_filter_val, epi2_filter_val, epi3_filter_val, epi4_filter_val, cutoff)
    test_filter_list = find_overlap(epi0_filter_test, epi1_filter_test, epi2_filter_test, epi3_filter_test, epi4_filter_test, cutoff)
    
    return train_filter_list, val_filter_list, test_filter_list

def filter_tensor(sequences_train_homsap, labels_train_homsap, sequences_val_homsap, labels_val_homsap, sequences_test_homsap, labels_test_homsap, percentile, cutoff):
    train_filter_list, val_filter_list, test_filter_list = \
    filter_idx(labels_train_homsap, labels_val_homsap, labels_test_homsap, percentile, cutoff)
    
    train_keep_list = [i for i in range(labels_train_homsap.size(0)) if i not in train_filter_list]
    val_keep_list = [i for i in range(labels_val_homsap.size(0)) if i not in val_filter_list]
    test_keep_list = [i for i in range(labels_test_homsap.size(0)) if i not in test_filter_list]
    
    sequences_train_homsap = sequences_train_homsap[train_keep_list]
    labels_train_homsap = labels_train_homsap[train_keep_list]

    sequences_val_homsap = sequences_val_homsap[val_keep_list]
    labels_val_homsap = labels_val_homsap[val_keep_list]

    sequences_test_homsap = sequences_test_homsap[test_keep_list]
    labels_test_homsap = labels_test_homsap[test_keep_list]
    
    return sequences_train_homsap, labels_train_homsap, sequences_val_homsap, labels_val_homsap, sequences_test_homsap, labels_test_homsap

sequences_train_homsap = torch.load('./sequences_train_homsap.pt')
labels_train_homsap = torch.load('./labels_train_homsap.pt')
sequences_val_homsap = torch.load('./sequences_val_homsap.pt')
labels_val_homsap = torch.load('./labels_val_homsap.pt')
sequences_test_homsap = torch.load('./sequences_test_homsap.pt')
labels_test_homsap = torch.load('./labels_test_homsap.pt')

labels_train_homsap = torch.cat((labels_train_homsap[:,:3,:],labels_train_homsap[:,4:,:]),dim=1)
labels_val_homsap = torch.cat((labels_val_homsap[:,:3,:],labels_val_homsap[:,4:,:]),dim=1)
labels_test_homsap = torch.cat((labels_test_homsap[:,:3,:],labels_test_homsap[:,4:,:]),dim=1)

sequences_train_homsap, labels_train_homsap, sequences_val_homsap, labels_val_homsap, sequences_test_homsap, labels_test_homsap = \
filter_tensor(sequences_train_homsap, labels_train_homsap, sequences_val_homsap, labels_val_homsap, sequences_test_homsap, labels_test_homsap, 5, 1)

sequences_train_macaca = torch.load('./sequences_train_macaca.pt')
labels_train_macaca = torch.load('./labels_train_macaca.pt')
sequences_val_macaca = torch.load('./sequences_val_macaca.pt')
labels_val_macaca = torch.load('./labels_val_macaca.pt')
sequences_test_macaca = torch.load('./sequences_test_macaca.pt')
labels_test_macaca = torch.load('./labels_test_macaca.pt')

labels_train_macaca = torch.cat((labels_train_macaca[:,:3,:],labels_train_macaca[:,4:,:]),dim=1)
labels_val_macaca = torch.cat((labels_val_macaca[:,:3,:],labels_val_macaca[:,4:,:]),dim=1)
labels_test_macaca = torch.cat((labels_test_macaca[:,:3,:],labels_test_macaca[:,4:,:]),dim=1)

sequences_train_macaca, labels_train_macaca, sequences_val_macaca, labels_val_macaca, sequences_test_macaca, labels_test_macaca = \
filter_tensor(sequences_train_macaca, labels_train_macaca, sequences_val_macaca, labels_val_macaca, sequences_test_macaca, labels_test_macaca, 10, 1)

sequences_train_gorgor = torch.load('./sequences_train_gorgor.pt')
labels_train_gorgor = torch.load('./labels_train_gorgor.pt')
sequences_val_gorgor = torch.load('./sequences_val_gorgor.pt')
labels_val_gorgor = torch.load('./labels_val_gorgor.pt')
sequences_test_gorgor = torch.load('./sequences_test_gorgor.pt')
labels_test_gorgor = torch.load('./labels_test_gorgor.pt')

labels_train_gorgor = torch.cat((labels_train_gorgor[:,:3,:],labels_train_gorgor[:,4:,:]),dim=1)
labels_val_gorgor = torch.cat((labels_val_gorgor[:,:3,:],labels_val_gorgor[:,4:,:]),dim=1)
labels_test_gorgor = torch.cat((labels_test_gorgor[:,:3,:],labels_test_gorgor[:,4:,:]),dim=1)

sequences_train_gorgor, labels_train_gorgor, sequences_val_gorgor, labels_val_gorgor, sequences_test_gorgor, labels_test_gorgor = \
filter_tensor(sequences_train_gorgor, labels_train_gorgor, sequences_val_gorgor, labels_val_gorgor, sequences_test_gorgor, labels_test_gorgor, 10, 1)

sequences_train_pantro = torch.load('./sequences_train_pantro.pt')
labels_train_pantro = torch.load('./labels_train_pantro.pt')
sequences_val_pantro = torch.load('./sequences_val_pantro.pt')
labels_val_pantro = torch.load('./labels_val_pantro.pt')
sequences_test_pantro = torch.load('./sequences_test_pantro.pt')
labels_test_pantro = torch.load('./labels_test_pantro.pt')

labels_train_pantro = torch.cat((labels_train_pantro[:,:3,:],labels_train_pantro[:,4:,:]),dim=1)
labels_val_pantro = torch.cat((labels_val_pantro[:,:3,:],labels_val_pantro[:,4:,:]),dim=1)
labels_test_pantro = torch.cat((labels_test_pantro[:,:3,:],labels_test_pantro[:,4:,:]),dim=1)

sequences_train_pantro, labels_train_pantro, sequences_val_pantro, labels_val_pantro, sequences_test_pantro, labels_test_pantro = \
filter_tensor(sequences_train_pantro, labels_train_pantro, sequences_val_pantro, labels_val_pantro, sequences_test_pantro, labels_test_pantro, 10, 1)

sequences_train_pongo = torch.load('./sequences_train_pongo.pt')
labels_train_pongo = torch.load('./labels_train_pongo.pt')
sequences_val_pongo = torch.load('./sequences_val_pongo.pt')
labels_val_pongo = torch.load('./labels_val_pongo.pt')
sequences_test_pongo = torch.load('./sequences_test_pongo.pt')
labels_test_pongo = torch.load('./labels_test_pongo.pt')

labels_train_pongo = torch.cat((labels_train_pongo[:,:3,:],labels_train_pongo[:,4:,:]),dim=1)
labels_val_pongo = torch.cat((labels_val_pongo[:,:3,:],labels_val_pongo[:,4:,:]),dim=1)
labels_test_pongo = torch.cat((labels_test_pongo[:,:3,:],labels_test_pongo[:,4:,:]),dim=1)

sequences_train_pongo, labels_train_pongo, sequences_val_pongo, labels_val_pongo, sequences_test_pongo, labels_test_pongo = \
filter_tensor(sequences_train_pongo, labels_train_pongo, sequences_val_pongo, labels_val_pongo, sequences_test_pongo, labels_test_pongo, 10, 1)


## Model
class GELU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    
class SoftmaxPooling1D(nn.Module):
    def __init__(self, pool_size=2, per_channel=False, w_init_scale=0.0):
        super(SoftmaxPooling1D, self).__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.w_init_scale = w_init_scale

    def forward(self, inputs):
        batch_size, channels, length = inputs.size()
        inputs = inputs.view(batch_size, channels, length // self.pool_size, self.pool_size)
        weights = torch.softmax(self.w_init_scale * inputs, dim=-1)
        return torch.sum(inputs * weights, dim=-1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        self.gelu = GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.pool(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.4):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
    
class TargetLengthCrop1D(nn.Module):
    def __init__(self, target_length):
        super(TargetLengthCrop1D, self).__init__()
        self.target_length = target_length
        
    def forward(self, inputs):
        if self.target_length is None:
            return inputs
        trim = (inputs.shape[-1] - self.target_length) // 2
        if trim < 0:
            raise ValueError('inputs longer than target length')
        elif trim == 0:
            return inputs
        else:
            return inputs[..., trim:-trim]
        
class Enformer(nn.Module):
    def __init__(self, channels=1408, num_transformer_layers=6, num_heads=8, pooling_type='attention'):
        super(Enformer, self).__init__()
        assert channels % num_heads == 0, "channels needs to be divisible by num_heads"

        self.stem = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=15, padding='same'),
            Residual(ConvBlock(256, 256, kernel_size=1, pool_size=1)),
            SoftmaxPooling1D(pool_size=2) if pooling_type == 'attention' else nn.MaxPool1d(pool_size=2)
        )

        filter_list = self.exponential_linspace_int(256, 512, num=7, divisible_by=32)
        conv_blocks = []
        for in_channels, out_channels in zip(filter_list[:-1], filter_list[1:]):
            conv_blocks.append(ConvBlock(in_channels, out_channels, kernel_size=20, pool_size=1))
            conv_blocks.append(Residual(ConvBlock(out_channels, out_channels, kernel_size=1, pool_size=1)))
            if pooling_type == 'attention':
                conv_blocks.append(SoftmaxPooling1D(pool_size=2))
            else:
                conv_blocks.append(nn.MaxPool1d(pool_size=2))
        self.conv_tower = nn.Sequential(*conv_blocks)
        transformer_blocks = [TransformerBlock(embed_dim=512, num_heads=num_heads) for _ in range(num_transformer_layers)]
        self.transformer = nn.Sequential(*transformer_blocks)

        self.crop_final = TargetLengthCrop1D(1152)
        self.final_pointwise = nn.Sequential(
            ConvBlock(512, 512, kernel_size=1, pool_size=1),
            nn.Dropout(0.3),
            GELU()
        )

        self.heads = nn.ModuleDict({
            'all_species': 
            nn.Sequential(
                nn.Conv1d(512, 5, kernel_size=1),
                nn.Softplus()
            )
        })
    
    def forward(self, x1):
        x1 = self.stem(x1)
        x1 = self.conv_tower(x1)
        '''
        x1 = x1.permute(2, 0, 1)
        x1 = self.transformer(x1)
        x1 = x1.permute(1, 2, 0).contiguous()
        '''
        x1 = self.crop_final(x1)
        x1 = self.final_pointwise(x1)
        output1 = self.heads['all_species'](x1)
        
        return output1

    @staticmethod
    def exponential_linspace_int(start, end, num, divisible_by=1):
        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)
        base = np.exp(np.log(end / start) / (num - 1))
        return [_round(start * base**i) for i in range(num)]

    
## Training model
def data_augmentation_flip(batch_sequences, batch_label):
    dim_size = batch_label.size(0)
    indices = torch.randperm(dim_size)
   
    half_size = dim_size // 2
    first_half_indices = indices[:half_size]
    second_half_indices = indices[half_size:]

    first_half_label = batch_label[first_half_indices]
    second_half_label = batch_label[second_half_indices]

    flipped_first_half_label = torch.flip(first_half_label, dims=[-1])
    new_batch_label = torch.cat((flipped_first_half_label, second_half_label), dim=0)
    
    first_half_sequence = batch_sequences[first_half_indices]
    second_half_sequence = batch_sequences[second_half_indices]
    
    flipped_first_half_sequence = torch.flip(first_half_sequence, dims=[-1])
    
    replacement_matrix = torch.tensor([
        [0., 0., 0., 1.],  # A -> T
        [0., 0., 1., 0.],  # C -> G
        [0., 1., 0., 0.],  # G -> C
        [1., 0., 0., 0.]   # T -> A
    ])
    RC_first_half_sequence = torch.matmul(flipped_first_half_sequence.permute(0, 2, 1), replacement_matrix).permute(0, 2, 1) 
    new_batch_sequence = torch.cat((RC_first_half_sequence, second_half_sequence), dim=0)
    return new_batch_sequence, new_batch_label


def data_augmentation_sub(batch_sequences, batch_label, region_length=1408):
    batch_size = batch_label.shape[0]
    
    start_indices = torch.randint(0, 1536 - region_length + 1, (batch_size,))
    batch_label_subs = []
    batch_sequences_subs = []
    for i in range(batch_size):
        start_idx = start_indices[i].item()
        batch_label_sub = batch_label[i, :, start_idx+128:start_idx+region_length-128]
        batch_label_subs.append(batch_label_sub)
        
        batch_sequences_sub = batch_sequences[i, :, start_idx*128:start_idx*128 + region_length*128]
        batch_sequences_subs.append(batch_sequences_sub)
        

    batch_label_subs_tensor = torch.stack(batch_label_subs)
    batch_sequences_subs_tensor = torch.stack(batch_sequences_subs)
    
    return batch_sequences_subs_tensor, batch_label_subs_tensor

def data_augmentation_shift(batch_sequences):
    for idx in range(batch_sequences.shape[0]):
        tmp = batch_sequences[idx]
        direction = random.choice(['left','right'])
        shift_bp = random.choice([0,1,2,3])
        zeros_tensor = torch.zeros(4, shift_bp)
        if direction == 'left':
            tmp = tmp[:,shift_bp:]
            new_tensor = torch.cat((tmp, zeros_tensor), dim=1)
        else:
            if shift_bp == 0:
                tmp = tmp
            else:
                tmp = tmp[:,:-shift_bp]
            new_tensor = torch.cat((zeros_tensor, tmp), dim=1)
        batch_sequences[idx] = new_tensor
    return batch_sequences

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
            
def train_model(model, criterion, optimizer, epochs, batch_size, label,
                sequences_train_human, labels_train_human, 
                sequences_train_macaca, labels_train_macaca,
                sequences_train_gorgor, labels_train_gorgor,
                sequences_train_pantro, labels_train_pantro,
                sequences_train_pongo, labels_train_pongo,
                sequences_val_human, labels_val_human, 
                sequences_val_macaca, labels_val_macaca,
                sequences_val_gorgor, labels_val_gorgor,
                sequences_val_pantro, labels_val_pantro,
                sequences_val_pongo, labels_val_pongo,
                if_sub=True, if_flip=True, if_shift=True):
    early_stopping = EarlyStopping(patience=5)
    
    batch_size = batch_size
    
    dataset_human = TensorDataset(sequences_train_human, labels_train_human)
    dataloader_human = DataLoader(dataset_human, batch_size=batch_size, shuffle=True)

    dataset_macaca = TensorDataset(sequences_train_macaca, labels_train_macaca)
    dataloader_macaca = DataLoader(dataset_macaca, batch_size=batch_size, shuffle=True)
    
    dataset_gorgor = TensorDataset(sequences_train_gorgor, labels_train_gorgor)
    dataloader_gorgor = DataLoader(dataset_gorgor, batch_size=batch_size, shuffle=True)
    
    dataset_pantro = TensorDataset(sequences_train_pantro, labels_train_pantro)
    dataloader_pantro = DataLoader(dataset_pantro, batch_size=batch_size, shuffle=True)
    
    dataset_pongo = TensorDataset(sequences_train_pongo, labels_train_pongo)
    dataloader_pongo = DataLoader(dataset_pongo, batch_size=batch_size, shuffle=True)
    
    
    train_loss = []
    val_loss = {}
    for epoch in range(epochs):  
        total_loss = 0.0
        for (inputs_task1, labels_task1),\
            (inputs_task2, labels_task2),\
            (inputs_task3, labels_task3),\
            (inputs_task4, labels_task4),\
            (inputs_task5, labels_task5)\
            in zip(dataloader_human, dataloader_macaca, dataloader_gorgor, dataloader_pantro, dataloader_pongo):
            
            if if_sub:
                inputs_task1, labels_task1 = data_augmentation_sub(inputs_task1, labels_task1)
                inputs_task2, labels_task2 = data_augmentation_sub(inputs_task2, labels_task2)
                inputs_task3, labels_task3 = data_augmentation_sub(inputs_task3, labels_task3)
                inputs_task4, labels_task4 = data_augmentation_sub(inputs_task4, labels_task4)
                inputs_task5, labels_task5 = data_augmentation_sub(inputs_task5, labels_task5)
            if if_flip:
                inputs_task1, labels_task1 = data_augmentation_flip(inputs_task1, labels_task1)
                inputs_task2, labels_task2 = data_augmentation_flip(inputs_task2, labels_task2)
                inputs_task3, labels_task3 = data_augmentation_flip(inputs_task3, labels_task3)
                inputs_task4, labels_task4 = data_augmentation_flip(inputs_task4, labels_task4)
                inputs_task5, labels_task5 = data_augmentation_flip(inputs_task5, labels_task5)
            if if_shift:
                inputs_task1 = data_augmentation_shift(inputs_task1)
                inputs_task2 = data_augmentation_shift(inputs_task2)
                inputs_task3 = data_augmentation_shift(inputs_task3)
                inputs_task4 = data_augmentation_shift(inputs_task4)
                inputs_task5 = data_augmentation_shift(inputs_task5)
            
            inputs_task = torch.cat([inputs_task1,inputs_task2,inputs_task3,inputs_task4,inputs_task5],dim=0)
            labels_task = torch.cat([labels_task1,labels_task2,labels_task3,labels_task4,labels_task5],dim=0)
            
            inputs_task = inputs_task.to(device)
            labels_task = labels_task.to(device)
                
            
            optimizer.zero_grad()
            outputs_task= \
            model(inputs_task)
            loss_task = criterion(outputs_task, labels_task)
            
            loss_task_all = loss_task
            loss_task_all.backward()
            optimizer.step()
            total_loss += loss_task_all.item()
    
        average_loss = total_loss / (len(dataloader_human)+len(dataloader_macaca)+len(dataloader_gorgor)+len(dataloader_pantro)+len(dataloader_pongo))
        train_loss.append(average_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train data, Loss: {average_loss}')
        
        
        ## validation
        val_loss[epoch+1] = {}
        val_num_homsap = int(np.floor(sequences_val_homsap.shape[0]/10))
        val_num_macaca = int(np.floor(sequences_val_macaca.shape[0]/10))
        val_num_gorgor = int(np.floor(sequences_val_gorgor.shape[0]/10))
        val_num_pantro = int(np.floor(sequences_val_pantro.shape[0]/10))
        val_num_pongo = int(np.floor(sequences_val_pongo.shape[0]/10))
        
        
        val_loss_human = []
        for i in range(val_num_homsap):
            with torch.no_grad():
                outputs_val_human = model(sequences_val_human[10*i:10*(i+1)])
                loss_val = criterion(outputs_val_human, labels_val_human[10*i:10*(i+1)])
            val_loss_human.append(loss_val.item())
        average_val_loss_human = np.mean(val_loss_human)
        
        val_loss_macaca = []
        for i in range(val_num_macaca):
            with torch.no_grad():
                outputs_val_macaca = model(sequences_val_macaca[10*i:10*(i+1)])
                loss_val = criterion(outputs_val_macaca, labels_val_macaca[10*i:10*(i+1)])
            val_loss_macaca.append(loss_val.item())
        average_val_loss_macaca = np.mean(val_loss_macaca)
        
        
        val_loss_gorgor = []
        for i in range(val_num_gorgor):
            with torch.no_grad():
                outputs_val_gorgor = model(sequences_val_gorgor[10*i:10*(i+1)])
                loss_val = criterion(outputs_val_gorgor, labels_val_gorgor[10*i:10*(i+1)])
            val_loss_gorgor.append(loss_val.item())
        average_val_loss_gorgor = np.mean(val_loss_gorgor)
        
        val_loss_pantro = []
        for i in range(val_num_pantro):
            with torch.no_grad():
                outputs_val_pantro = model(sequences_val_pantro[10*i:10*(i+1)])
                loss_val = criterion(outputs_val_pantro, labels_val_pantro[10*i:10*(i+1)])
            val_loss_pantro.append(loss_val.item())
        average_val_loss_pantro = np.mean(val_loss_pantro)
        
        val_loss_pongo = []
        for i in range(val_num_pongo):
            with torch.no_grad():
                outputs_val_pongo = model(sequences_val_pongo[10*i:10*(i+1)])
                loss_val = criterion(outputs_val_pongo, labels_val_pongo[10*i:10*(i+1)])
            val_loss_pongo.append(loss_val.item())
        average_val_loss_pongo = np.mean(val_loss_pongo)
        
        val_loss[epoch+1]['human'] = average_val_loss_human
        val_loss[epoch+1]['macaca'] = average_val_loss_macaca
        val_loss[epoch+1]['gorgor'] = average_val_loss_gorgor
        val_loss[epoch+1]['pantro'] = average_val_loss_pantro
        val_loss[epoch+1]['pongo'] = average_val_loss_pongo
        
        average_val_loss = (average_val_loss_human + average_val_loss_macaca + average_val_loss_gorgor + average_val_loss_pantro + average_val_loss_pongo)/5
        
        print(f'Epoch [{epoch+1}/{epochs}], Validation data, Loss Human: {average_val_loss_human}, Loss Macaca: {average_val_loss_macaca}, Loss GorGor: {average_val_loss_gorgor}, Loss PanTro: {average_val_loss_pantro}, Loss Pongo: {average_val_loss_pongo}')
        print(f'Epoch [{epoch+1}/{epochs}], Validation data, Loss All: {average_val_loss}')
        torch.save(model, './Model/pretrain_model_allspecies_kernel15_CNN_'+label+'_epoch'+str(epoch+1)+'.pth')
        
        early_stopping(average_val_loss, epoch+1)
        if early_stopping.early_stop:
            print("Early stopping")
            print(early_stopping.best_epoch, early_stopping.best_loss)
            break
        
    return train_loss, val_loss

from torch.nn.parallel import DataParallel

model = Enformer()
device = torch.device('cuda:0')
model = DataParallel(model, device_ids=[0,5,6])
model = model.to(device)

labels_val_macaca = labels_val_macaca.to(device)
labels_val_homsap = labels_val_homsap.to(device)
labels_val_gorgor = labels_val_gorgor.to(device)
labels_val_pantro = labels_val_pantro.to(device)
labels_val_pongo = labels_val_pongo.to(device)

sequences_val_homsap_sub = sequences_val_homsap[:,:,(196608-180224)//2:(196608-180224)//2+180224]
sequences_val_macaca_sub = sequences_val_macaca[:,:,(196608-180224)//2:(196608-180224)//2+180224]
sequences_val_gorgor_sub = sequences_val_gorgor[:,:,(196608-180224)//2:(196608-180224)//2+180224]
sequences_val_pantro_sub = sequences_val_pantro[:,:,(196608-180224)//2:(196608-180224)//2+180224]
sequences_val_pongo_sub = sequences_val_pongo[:,:,(196608-180224)//2:(196608-180224)//2+180224]

labels_val_homsap_sub = labels_val_homsap[:,:,(1536-1152)//2:(1536-1152)//2+1152]
labels_val_macaca_sub = labels_val_macaca[:,:,(1536-1152)//2:(1536-1152)//2+1152]
labels_val_gorgor_sub = labels_val_gorgor[:,:,(1536-1152)//2:(1536-1152)//2+1152]
labels_val_pantro_sub = labels_val_pantro[:,:,(1536-1152)//2:(1536-1152)//2+1152]
labels_val_pongo_sub = labels_val_pongo[:,:,(1536-1152)//2:(1536-1152)//2+1152]

criterion = nn.PoissonNLLLoss(log_input=False,full=True)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

epochs = 100
batch_size = 8
label = 'onehead_PoissionNLLLoss'
train1_loss, val_loss = \
train_model(model, criterion, optimizer, epochs, batch_size, label,
                sequences_train_homsap, labels_train_homsap, 
                sequences_train_macaca, labels_train_macaca,
                sequences_train_gorgor, labels_train_gorgor,
                sequences_train_pantro, labels_train_pantro,
                sequences_train_pongo, labels_train_pongo,
                sequences_val_homsap_sub, labels_val_homsap_sub, 
                sequences_val_macaca_sub, labels_val_macaca_sub,
                sequences_val_gorgor_sub, labels_val_gorgor_sub,
                sequences_val_pantro_sub, labels_val_pantro_sub,
                sequences_val_pongo_sub, labels_val_pongo_sub,
                if_sub=True, if_flip=True, if_shift=True)