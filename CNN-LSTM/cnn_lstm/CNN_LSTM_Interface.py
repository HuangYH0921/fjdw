import tensorflow as tf
from  tensorflow.keras import Sequential
from  tensorflow.keras.layers import LSTM,Dense,Activation,Dropout
from  tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,RepeatVector
from  tensorflow.keras.callbacks import History,EarlyStopping
import  numpy as np
import os
import random

def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# seed_tensorflow(42)

def create_dataset(dataset, n_predictions,next_num):
    '''
    对数据进行处理
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - n_predictions - next_num + 1):
        dataX.append(dataset[i:(i+n_predictions),:])
        dataY.append(dataset[(i+n_predictions):(i+n_predictions+next_num),:])
    train_x = np.array(dataX)
    train_y = np.array(dataY)
    return train_x, train_y


#多维归一化
def Normalize_Mult(data):
    normalize = np.zeros((data.shape[1],2),dtype='float64')
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,:] = [listlow,listhigh]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta 
    return  data,normalize


def FNormalize(data, norm):
    listlow,listhigh= norm[0],norm[1]
    delta = listhigh - listlow
    if delta != 0:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = data[i, j] * delta + listlow
    return data


def cnn_lstm_model(train_x,train_y,config):

    model = Sequential()
    
    model.add(Conv2D(
        filters = config.nb_filter[0], 
        kernel_size = config.kernel_size,
        activation = ('relu'), 
        padding= 'SAME',
        input_shape = (config.n_predictions,config.features,1),
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)))
    
    model.add(MaxPooling2D(pool_size=config.pool_length))
    
    model.add(Conv2D(
        filters = config.nb_filter[1], 
        kernel_size = config.kernel_size,
        activation = ('relu'),
        padding= 'SAME'))

    model.add(Flatten())
    model.add(RepeatVector(train_y.shape[1]))

    model.add(LSTM(config.lstm_layers, return_sequences=False))
    # model.add(LSTM(config.lstm_layers,input_shape=(train_x.shape[1],train_x.shape[2]),
    #                return_sequences=False))
    model.add(Dropout(config.dropout))
    
    model.add(Dense(train_y.shape[1]))
    model.add(Activation("relu"))

    model.summary()

    cbs = [History(), EarlyStopping(monitor='val_loss',
                                    patience=config.patience,
                                    min_delta=config.min_delta,
                                    verbose=0)]
    model.compile(loss=config.loss_metric,optimizer=config.optimizer)
    model.fit(
        train_x,
        train_y,
        batch_size = config.lstm_batch_size,
        epochs = config.epochs,
        validation_split = config.validation_split,
        callbacks = cbs,
        verbose = config.verbose,
        )
    return model


def start_Train(data,config):
    seed_tensorflow(42)
    #删去时间步
    data = data.iloc[:, 1:]
    print(data.columns)

    yindex = data.columns.get_loc(config.dimname)
    data = np.array(data, dtype='float64')

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # 数据归一化
    data, normalize = Normalize_Mult(data)
    data_y = data[:, yindex]
    if len(data_y.shape) == 1:
        data_y = data_y.reshape(data_y.shape[0], 1)

    # 构造训练数据
    train_x, _ = create_dataset(data, config.n_predictions, config.next_num)
    _, train_y = create_dataset(data_y, config.n_predictions, config.next_num)
    print(train_x.shape)
    # 2D卷积
    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)

    # 进行训练
    model = cnn_lstm_model(train_x,train_y,config)

    return  model,normalize