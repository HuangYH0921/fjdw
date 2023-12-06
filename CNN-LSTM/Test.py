import  pandas as pd
import  numpy as np
from keras.models import  load_model
from cnn_lstm.Predict_Interface import Predict
from base import get_list_element

def Test(data,config,numrows):
    # numrows = data.iloc[:,1].notnull().sum()  # 已知电量数据的天数
    ind = [['日期'],[config.dimname],config.covariance]
    ind = get_list_element(ind)
    data = data.loc[:,ind]
    
    data = data.iloc[:numrows + config.n_predictions,:]
    # 如果电量数据不足，则用第一条数据向前填充
    if numrows - config.n_predictions < 0:
        data = pd.concat([pd.DataFrame(-(numrows - config.n_predictions)*[[np.NaN]*len(data.columns)],columns = data.columns),data],ignore_index=True)
        data[config.dimname] = data[config.dimname].fillna(method = 'bfill')
        numrows = config.n_predictions
    data = data.iloc[numrows - config.n_predictions:,:]  #  取出用于预测的数据
    
    # # 提取协变量对应的字段
    
    
    if '平均气温' in config.covariance:
        data.insert(2, '平均气温1', data['平均气温'])
        data['平均气温1'] = data['平均气温1'].shift(config.next_num)
        
    if '最高气温' in config.covariance:
        data.insert(len(data.columns), '最高气温1', data['最高气温'])
        data['最高气温1'] = data['最高气温1'].shift(config.next_num)
        
    data=data.fillna(0)
    data[config.dimname] = data[config.dimname].shift(config.next_num)
    data = data.dropna(axis=0,how='any')
    data[config.event] = 0   #假如事件不存在, 把相应协变量改为0
    data.reset_index(drop=True, inplace=True)

    # 读取权重数据和归一化数据
    normalize = np.load(config.path+config.city+config.dimname+".npy")
    model = load_model(config.path+config.city+config.dimname+".h5")
    
    hat_y = Predict(model, data, normalize, config)
    hat_y = np.array(hat_y, dtype='float64').reshape(-1,1)
    day_result = hat_y[-config.next_num:]
    
    sum_result = np.sum(day_result)
   
    return sum_result,day_result
