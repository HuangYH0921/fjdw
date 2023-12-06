#使用类实现一个配置文件
class Config:
    def __init__(self, city, Category, n, event):
        
        self.city = city
        self.path = './Model/' # 存储权重、归一化数据的路径
        
        self.event = event
        
        # 要预测的列
        self.dimname = Category
        #协变量有星期，是否节假日，节日，平均气温，最高气温，最低气温
        if self.dimname == "大工业":
            self.covariance = ['星期','是否节假日','节日','疫情','春节','国庆']

        if self.dimname == "非普":
            self.covariance = ['是否节假日','疫情','春节','国庆']
            
        if self.dimname == "居民":
            self.covariance = ['平均气温','最高气温','疫情','春节','国庆']
            
        if self.dimname == "非居照":
            self.covariance = ['星期','是否节假日','平均气温','疫情','春节','国庆']
        
        if self.dimname == "商业":
            self.covariance = ['星期','平均气温','疫情','春节','国庆']
            
        if self.dimname == '农业':
            self.covariance = ['平均气温','疫情','春节','国庆']
            
        if self.dimname not in ['大工业','非普','居民','非居照','商业','农业']:
            self.covariance = ['是否节假日','疫情','春节','国庆']
            
        # 特征数
        self.features = len(self.covariance) + 1
        if '星期' in self.covariance:
            self.features = self.features + 5
            self.covariance = [['星期三','星期二','星期五','星期六','星期四','星期日'] 
                               if i == '星期' else i for i in self.covariance]
        if '平均气温' in self.covariance:
            self.features = self.features + 1
            
        if '最高气温' in self.covariance:
            self.features = self.features + 1
        
        # 预测接下来几个时间步的值
        self.next_num = n
        # 使用前n_predictions 步去预测下一步(滑窗大小）
        # self.n_predictions = 2 * self.next_num
        self.n_predictions = self.next_num
        
        #指定EarlyStopping  如果训练时单次val_loss值不能至少减少min_delta时，最多允许再训练patience次
        #能够容忍多少个epoch内都没有improvement
        self.patience = 20 #--10
        self.min_delta = 0.00001


        #2维卷积层参数
        self.nb_filter = [32,64]     #滤波器个数
        self.kernel_size = (3,3)     #滤波器尺寸
        self.pool_length = (2,2)     #池化长度

        #指定LSTM神经元个数
        self.lstm_layers = 128
        self.dropout = 0.2

        self.lstm_batch_size = 64
        self.optimizer = 'adam'
        self.loss_metric = 'mse'
        self.validation_split = 0.2
        self.verbose = 1
        self.epochs = 200

