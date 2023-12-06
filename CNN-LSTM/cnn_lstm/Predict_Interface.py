import numpy as np
from cnn_lstm.CNN_LSTM_Interface import FNormalize

# 使用训练数据的归一化
def Normalize_Data(data, normalize):
    for i in range(0, data.shape[1]):
        # 第i列
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            # 第j行
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    return data


def Predict(model, data, normalize, config):
    # 多步预测
    # 删去时间列
    data = data.iloc[:, 1:]
    yindex = data.columns.get_loc(config.dimname)

    data = np.array(data, dtype='float64')
    # 使用训练数据边界进行的归一化
    data = Normalize_Data(data, normalize)

    test_x = data[:config.n_predictions,:]
    test_x = test_x.reshape(1,test_x.shape[0],test_x.shape[1],1)
    # test_x = test_x.reshape(1,test_x.shape[0],test_x.shape[1])
    # print(test_x.shape)

    # 加载模型
    hat_y = model.predict(test_x)

    # 反归一化
    hat_y = FNormalize(hat_y, normalize[yindex,])
    return hat_y

