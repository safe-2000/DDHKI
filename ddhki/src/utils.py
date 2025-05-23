import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable



def dynamic_temperature(epoch, max_epochs, init_temp=0.5, min_temp=0.1):
    """
    动态调整温度，随着训练进程逐渐降低。
    """
    return max(min_temp, init_temp * (1 - epoch / max_epochs))

def hard_negative_mining(similarity_matrix, labels, margin=0.1):
    """
    选择难负样本，只有与正样本相似度接近的负样本参与损失计算。
    """
    batch_size = similarity_matrix.size(0)
    
    # 获取正样本相似度（对角线上的值）
    positive_sim = similarity_matrix[torch.arange(batch_size), labels]
    
    # 负样本相似度矩阵
    negative_sim = similarity_matrix.clone()
    negative_sim[torch.arange(batch_size), labels] = float('-inf')  # 排除正样本
    
    # 选择比正样本相似度高出 margin 的难负样本
    hard_negatives = negative_sim > (positive_sim.unsqueeze(1) - margin)
    
    return hard_negatives

def enhanced_info_nce_loss(augmented, original, epoch, max_epochs, temperature=0.5, margin=0.1):
    batch_size = augmented.size(0)
    
    # 动态调整温度
    temperature = dynamic_temperature(epoch, max_epochs, init_temp=temperature)
    
    # 归一化向量
    augmented_norm = F.normalize(augmented, dim=1)
    original_norm = F.normalize(original, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.mm(augmented_norm, original_norm.T) / temperature
    
    # 生成标签，正例的标签是对角线上的
    labels = torch.arange(batch_size).to(original.device)
    
    # 进行难负样本挖掘
    hard_negatives = hard_negative_mining(similarity_matrix, labels, margin=margin)
    
    # 只对难负样本计算损失
    similarity_matrix[~hard_negatives] = float('-inf')
    
    # 计算损失
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss
