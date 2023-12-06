import os
import pandas as pd
import numpy as np
import xlrd

def change_temperature(data, city):
    # 各城市对应的温度变点
    Variable_point = [18.0572438577213, 25.433784241020590, 24.462820756558070, 24.4848, 25.449853697035508,
                      23.3827, 24.843420478896853, 25.306678569162464,24.711891879695347]
    left_slope = [-0.0846, -0.001862515, -0.003864866, -9.60E-04, -0.002678598,	
                  -0.0029, -0.001728595, -0.002076953, -0.00254705]
    right_slope = [0.0574, 0.01432445, 0.034660831, 0.0293, 0.020517871,
                   0.0039, 0.00476996, 0.006416967, 0.011393384]
    cities = ["福州","莆田","泉州","厦门","漳州","龙岩","三明","南平","宁德"]
    ind = cities.index(city)
    
    vp = Variable_point[ind]
    ls = left_slope[ind]
    rs = right_slope[ind]
    change_data = data.mask(data < vp, (data-vp)*ls)
    change_data = change_data.mask(data >= vp, (data-vp)*rs)
    
    return change_data

def excel2csv(filepath,savepath):
    work_book = xlrd.open_workbook(filepath)
    sheetName = work_book.sheet_names()

    folder = os.path.exists(savepath)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(savepath)

    for sheetname in sheetName:
        df = pd.read_excel(filepath, sheet_name=sheetname)
        df.to_csv(savepath + sheetname+'.csv', encoding='gbk',index=None)

def removefile(filepath):   
    if(os.path.isfile(filepath)):
        os.remove(filepath)
        print("File Deleted successfully")
    else:
        print("File does not exist")
        
def get_list_element(lis):
    result_list = []
    for i in lis:
        if type(i) == list:
            result_list += get_list_element(i)
        else:
            result_list.append(i)
    return result_list


def OPE(y_hat,y_test):
    err = np.abs(np.sum(y_hat) - np.sum(y_test) / np.sum(y_test))
    return err

def MSE(y_hat,y_test):
    err = np.mean((y_hat- y_test)**2)
    return err

def MAE(y_hat,y_test): 
    err = np.mean(np.abs((y_hat- y_test)))
    return err

def MAPE(y_hat,y_test):
    err = np.mean(np.abs((y_hat - y_test) / y_test))
    return err

def WAPE(y_hat,y_test):
    err = np.sum(np.abs(y_hat- y_test))/np.sum(np.abs(y_test))
    return err

def RMSE(y_hat,y_test):
    err = np.mean((y_hat- y_test)**2)**0.5
    return err

def mkdir_multi(path):
    isExists=os.path.exists(path)
    if not isExists:
        # 如果不存在，则创建目录（多层）
        os.makedirs(path) 
        print('目录创建成功！')
        return True
    else:
        return False
        
    

     