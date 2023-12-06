import numpy as np
import pandas as pd
import os
from cnn_lstm.CNN_LSTM_Interface import start_Train
from base import get_list_element

os.environ['CUDA_VISIBLE_DEVICES']='0'


def Train(data,config):
    
    if not os.path.exists(config.path): os.makedirs(config.path)
    
    data = data.iloc[:data.iloc[:,1].notnull().sum() + config.next_num,:]
    # 提取协变量对应的字段
    ind = [['日期'],[config.dimname],config.covariance]
    ind = get_list_element(ind)
    data = data.loc[:,ind]

    # 用到预测日对应的气温数据(涉及到数据穿越问题)
    if '平均气温' in config.covariance:
        data.insert(2, '平均气温1', data['平均气温'])
        data['平均气温1'] = data['平均气温1'].shift(config.next_num)
    
    if '最高气温' in config.covariance:
        data.insert(len(data.columns), '最高气温1', data['最高气温'])
        data['最高气温1'] = data['最高气温1'].shift(config.next_num)
        
    data=data.fillna(0)
    data[config.dimname] = data[config.dimname].shift(config.next_num)
    data = data.dropna(axis=0,how='any')
    
    model,normalize = start_Train(data,config)
    
    model.save(config.path+config.city+config.dimname+".h5")  # 保存权重数据
    np.save(config.path+config.city+config.dimname+".npy",normalize) # 保存归一化的listlow,listhigh