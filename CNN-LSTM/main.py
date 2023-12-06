from Config import Config
from Train import Train
from Test import Test
import pandas as pd
import numpy as np
import shutil
from base import excel2csv,removefile,change_temperature, mkdir_multi
import os
import pyecharts.options as opts
from pyecharts.charts import Line,Page
    

def train_part(cities,Categories,n,filepath,event):
    
    for i in cities:
        n0 = n[i]
        data = pd.read_excel(filepath, sheet_name=i)
        need_ind = ['日期'] + Categories + ['星期','最高气温','是否节假日','节日','平均气温','疫情','春节','国庆'] #需要用到的字段
        data = data.loc[:,need_ind]
        # 有节日的设为1，否则为0
        data["节日"] = data["节日"].where(data["节日"].isnull(), 1)
        data["节日"] = data["节日"].fillna(0)
        # 对星期进行虚拟编码
        data = data.join(pd.get_dummies(data.星期, drop_first=True))  
        data = data.drop(['星期'],axis=1)
        
        for j in Categories:
            print('----------{} {} 训练模型---------'.format(i,j))
            config = Config(i,j, n0, event)
            Train(data,config)
    
def pred_part(n,cities,Categories,preDate,filepath,storepath,event):
    
    for i in cities:
        n0 = n[i]
        writer = pd.ExcelWriter(storepath + i + '不存在' + event +'.xlsx') 
        data = pd.read_excel(filepath, sheet_name=i)
        need_ind = ['日期'] + Categories + ['星期','最高气温','是否节假日','节日','平均气温','疫情','春节','国庆']
        data = data.loc[:,need_ind]
        # 有节日的设为1，否则为0
        data["节日"] = data["节日"].where(data["节日"].isnull(), 1)
        data["节日"] = data["节日"].fillna(0)
        # 对星期进行虚拟编码
        data = data.join(pd.get_dummies(data.星期, drop_first=True))  
        data = data.drop(['星期'],axis=1)
        
        for k in preDate[i]:
            # 预测
            numrows = data.loc[data["日期"]==k].index[0]
            day_result = []
            for j in Categories:
                print('----------{} {} {} 预测---------'.format(k,i,j))
                config = Config(i,j,n0,event)
                _, d_pred = Test(data,config,numrows)
                day_result = np.append(day_result, d_pred)
                print("----------预测成功----------\n")
            
            day_result = pd.DataFrame(day_result.reshape(len(Categories), -1))
            day_result.index = Categories
            day_result.columns = pd.date_range(start=k, periods = n0, freq="D")
            day_result.to_excel(writer,k)
            
        writer.save() #文件保存
        writer.close() #文件关闭
        
def main(filepath,storepath,n,preDate,event):

    cities = list(preDate.keys())
    Categories = ['大工业','非普','居民','非居照','商业','农业']
    
    train_part(cities,Categories,n,filepath,event)
    pred_part(n,cities,Categories,preDate,filepath,storepath,event)
    
def main_hy(filepath,storepath,hy,n,preDate,event):
    
    cities = list(preDate.keys())
    
    train_part(cities, hy, n, filepath, event)
    pred_part(n, cities, hy, preDate, filepath, storepath, event)
    
    
def plot(city,keepTime,FilePath,TruePath,StorePath,event):
    data1 = pd.read_excel(FilePath + city + '不存在'+ event +'.xlsx',sheet_name=None)
    sheetName = list(data1.keys())
    data2 = pd.read_excel(TruePath, sheet_name = city)
    p = Page()
    impact_sum = np.array([])
    date_list = []
    record_i = {}
    record = pd.DataFrame()
    for i in range(len(sheetName)):
        data11 = data1[sheetName[i]]
        categories = data11.iloc[:,0].tolist()
        time = data11.columns[1:]
        t = pd.date_range(start=time[0],periods=1).strftime("%Y%m%d").tolist() #将日期格式化，以便keepTime查找
        x = pd.date_range(start=time[0], periods=keepTime[t[0]]).strftime("%Y%m%d").tolist()
        date_list.append(x[0]+'~'+x[-1])
        start = data2.loc[data2["日期"] == time[0]].index[0]
        data22 = data2.iloc[start:start + keepTime[t[0]],:]
        
        for j in range(len(categories)):
            y1 = data11.iloc[j,1:keepTime[t[0]]+1].tolist()
            y2 = data22[categories[j]]
            impact = (sum(y2)-sum(y1))/sum(y1)*100
            record_i = {"地市":city, event+"时间":x[0]+'——'+x[-1], "类别":categories[j], "电量受影响时间":x[0]+'——'+x[-1], "实际电量": sum(y2), "还原电量":sum(y1), "影响电量":sum(y1)-sum(y2) }
            record = record.append(record_i, ignore_index=True)
            impact_sum = np.append(impact_sum, impact)
            line=(
                Line()
                .add_xaxis(xaxis_data=x)
                .add_yaxis(
                    series_name="假如" + event + "不存在的用电量",
                    y_axis=y1,
                    is_smooth=True,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                )
                .add_yaxis(
                    series_name = event + "期间真实用电量",
                    y_axis=y2,
                    is_smooth=True,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                    z = 1
            
                )
   
            
                .set_global_opts(
                        title_opts=opts.TitleOpts(title=city + categories[j]+x[0]+'~'+x[-1],
                                                  subtitle = event+"对其综合影响:{:.3f}%".format(impact)),
                        
                        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                        xaxis_opts=opts.AxisOpts(boundary_gap=False),
                        yaxis_opts=opts.AxisOpts(
                            # axislabel_opts=opts.LabelOpts(formatter="{value} W"),
                            splitline_opts=opts.SplitLineOpts(is_show=True),
                        ),
                        
                        datazoom_opts=[opts.DataZoomOpts(type_="inside"),
                                       opts.DataZoomOpts(orient='horizontal')] # 坐标轴进行缩放
                    )
            
                )
            p.add(line)
            
        
    p.render(StorePath + city + event + '.html')
    
    impact_sum = pd.DataFrame(impact_sum.reshape(-1,len(categories)))
    impact_sum.columns = categories
    impact_sum.index = date_list
    
    return impact_sum, record
    

    
def plot_hy(city,keepTime,FilePath,TruePath,StorePath,event):
    data1 = pd.read_excel(FilePath + city + '不存在' + event + '.xlsx',sheet_name=None)
    sheetName = list(data1.keys())
    data2 = pd.read_excel(TruePath, sheet_name = city)
    p = Page()
    impact_sum = np.array([])
    date_list = []
    record_i = {}
    record = pd.DataFrame()
    for i in range(len(sheetName)):
        data11 = data1[sheetName[i]]
        categories = data11.iloc[:,0].tolist()
        time = data11.columns[1:]
        t = pd.date_range(start=time[0],periods=1).strftime("%Y%m%d").tolist() #将日期格式化，以便keepTime查找
        x = pd.date_range(start=time[0], periods=keepTime[t[0]]).strftime("%Y%m%d").tolist()
        date_list.append(x[0]+'~'+x[-1])
        start = data2.loc[data2["日期"] == time[0]].index[0]
        data22 = data2.iloc[start:start + keepTime[t[0]],:]
        for j in range(len(categories)):
            y1 = data11.iloc[j,1:keepTime[t[0]]+1].tolist()
            y2 = data22[categories[j]]
            impact = (sum(y2)-sum(y1))/sum(y1)*100
            record_i = {"地市":city, event+"时间":x[0]+'——'+x[-1], "类别":categories[j], "电量受影响时间":x[0]+'——'+x[-1], "实际电量": sum(y2), "还原电量":sum(y1), "影响电量":sum(y1)-sum(y2) }
            record = record.append(record_i, ignore_index=True)
            impact_sum = np.append(impact_sum, impact)
            
            line=(
                Line()
                .add_xaxis(xaxis_data=x)
                .add_yaxis(
                    series_name="假如"+event+"不存在的用电量",
                    y_axis=y1,
                    is_smooth=True,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                )
                .add_yaxis(
                    series_name=event+"期间真实用电量",
                    y_axis=y2,
                    is_smooth=True,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                    z = 1
            
                )
   

                .set_global_opts(
                        title_opts=opts.TitleOpts(title=city + categories[j]+x[0]+'~'+x[-1],
                                                  subtitle=event+"对其综合影响:{:.3f}%".format(impact)),
                        
                        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                        xaxis_opts=opts.AxisOpts(boundary_gap=False),
                        yaxis_opts=opts.AxisOpts(
                            # axislabel_opts=opts.LabelOpts(formatter="{value} W"),
                            splitline_opts=opts.SplitLineOpts(is_show=True),
                        ),
                        
                        datazoom_opts=[opts.DataZoomOpts(type_="inside"),
                                       opts.DataZoomOpts(orient='horizontal')] # 坐标轴进行缩放
                    )
            
                )
            p.add(line)
            
        
    p.render(StorePath + city + event + '.html')
    
    impact_sum = pd.DataFrame(impact_sum.reshape(-1,len(categories)))
    impact_sum.columns = categories
    impact_sum.index = date_list
    
    return impact_sum,record
    
if __name__ == '__main__':
    # ==================== 六个用电类别 ====================
    readDataPath = './data/九地市分用电类别数据(整理版).xlsx'
    storepath = './预测结果/春节影响/六个用电类别/'
    mkdir_multi(storepath)
    figurepath = './预测结果/春节影响/结果图/六个用电类别/'
    mkdir_multi(figurepath)
    data = pd.read_excel(readDataPath)
    feature = data.columns.tolist()[1:7]
    
    
    # ==================== 31个制造业 =====================
    # readDataPath = './data/31个制造业.xlsx'
    # storepath = './预测结果/国庆影响/31个制造业/'
    # mkdir_multi(storepath)
    # figurepath = './预测结果/国庆影响/结果图/31个制造业/'
    # mkdir_multi(figurepath)
    # data = pd.read_excel(readDataPath)
    # feature = data.columns.tolist()[1:33]
    
    
    # ================ 三大产业和十一大行业 ================
    # readDataPath = './data/三大产业和十一大行业.xlsx'
    # storepath = './预测结果/国庆影响/三大产业和十一大行业/'
    # mkdir_multi(storepath)
    # figurepath = './预测结果/国庆影响/结果图/三大产业和十一大行业/'
    # mkdir_multi(figurepath)
    # data = pd.read_excel(readDataPath)
    # feature = data.columns.tolist()[1:15]
    
    
    # ------------------ 研究疫情影响 ------------------
    # 六个用电类别
    # n = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],28) 
    # 三大产业、十一大行业、31制造业
    # n={}
    # n["厦门"]=26
    # n["福州"]=26
    # n["莆田"]=26
    # n["泉州"]=23
    # n["漳州"]=26
    # n["龙岩"]=26
    # n["三明"]=23
    # n["南平"]=23
    # n["宁德"]=26
    
    # keepTime= {}
    # keepTime["厦门"] = {"20200129":12, "20210913":23, "20220127":26, "20220913":7, "20221205":4}
    # keepTime["福州"] = {"20200124":17,"20221023":8}
    # keepTime["莆田"] = {"20200128":15,"20210911":26,"20220318":7}
    # keepTime["泉州"] = {"20200127":14,"20210912":15,"20220313":7}
    # keepTime["漳州"] = {"20200124":17,"20220318":7}
    # keepTime["龙岩"] = {"20200124":17}
    # keepTime["三明"] = {"20200127":14,"20220316":7}
    # keepTime["南平"] = {"20200127":14,"20220316":7}
    # keepTime["宁德"] = {"20200124":17,"20220412":7}
    
    # preDate = {}
    # preDate["厦门"] = ["20200129","20210913","20220127","20220913","20221205"]
    # preDate["福州"] = ["20200124","20221023"]
    # preDate["莆田"] = ["20200128","20210911","20220318"]
    # preDate["泉州"] = ["20200127","20210912","20220313"]
    # preDate["漳州"] = ["20200124","20220318"]
    # preDate["龙岩"] = ["20200124"]
    # preDate["三明"] = ["20200127","20220316"]
    # preDate["南平"] = ["20200127","20220316"]
    # preDate["宁德"] = ["20200124","20220412"]
    
    # ------------------ 研究春节影响 ------------------
    n = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],10)
    # n = dict.fromkeys(["福州"],69)
    
    # 六个用电类别
    keepTime = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],
                              {"20190204":7, "20200124":10, "20210211":7, "20220131":7, "20230121":7})
    
    # keepTime = dict.fromkeys(["福州"],
    #                           {"20191224":69, "20210111":65, "20220101":64, "20230101":27})
    
    # 三大产业、十一大行业、31制造业
    # keepTime = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],
    #                           {"20200124":10, "20210211":7, "20220131":7, "20230121":7})
    
    preDate = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],
                            list(keepTime["厦门"].keys()))
    # preDate = dict.fromkeys(["福州"],
    #                         list(keepTime["福州"].keys()))
    
    # ------------------ 研究国庆影响 ------------------
    # n = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],8)
    
    # # 六个用电类别
    # # keepTime = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],
    # #                           {"20191001":7, "20201001":8, "20211001":7, "20221001":7})
    
    # # 三大产业、十一大行业、31制造业
    # keepTime = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],
    #                           {"20201001":8, "20211001":7, "20221001":7})
    
    # preDate = dict.fromkeys(["厦门","福州","莆田","泉州","漳州","龙岩","三明","南平","宁德"],
    #                         list(keepTime["厦门"].keys()))
    
    
    # ------------------- 训练、预测 -------------------
    event = "春节"  # 研究的事件
    cities = list(preDate.keys())
    train_part(cities,feature,n,readDataPath,event)
    pred_part(n,cities,feature,preDate,readDataPath,storepath,event)
    
    
    
    
    
    
    
    # ==================== 画图 and 保存excel文件 ======================
    writer1 = pd.ExcelWriter(storepath + event + '影响.xlsx') 
    R = pd.DataFrame()
    for i in cities:
        _,record = plot(i,keepTime[i],storepath,readDataPath,figurepath,event)
        # _,record = plot_hy(i,keepTime[i],storepath,readDataPath,figurepath,event)
        R = pd.concat([R,record])

        
    R.reset_index(drop=True,inplace=True)
    R.to_excel(writer1)
    writer1.save() #文件保存
    writer1.close() #文件关闭
    