import os
import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Datasets import GeoSpatialDataset  # 确保 Datasets.py 位于同一目录或正确的Python路径中
from utils import evaluate_model_metrics  # 如果需要，可以使用 utils 中的功能
from Nets import TransformerRegressor
from tqdm import tqdm
import pickle  # 用于保存 scaler 和 feature list
import Debug
# 1. 加载预处理所需的 scaler 和特征列表
# from torchsummary import summary  # 新增
from concurrent.futures import ProcessPoolExecutor, as_completed
from accelerate import Accelerator

def load_scalers(dynamic_scaler_path='Deep_scaler_X_d.pkl', static_scaler_path='Deep_scaler_X_s.pkl',scaler3="Deep_scaler_X_y.pkl"):
    with open(dynamic_scaler_path, 'rb') as f:
        dynamic_scaler = pickle.load(f)
    with open(static_scaler_path, 'rb') as f:
        static_scaler = pickle.load(f)
        
    with open(scaler3, 'rb') as f:
        scaler3ls = pickle.load(f)
    return dynamic_scaler, static_scaler,scaler3ls

def load_model(model_path, device):
    accelerator = Accelerator()  # 初始化 Accelerator
    model = TransformerRegressor(
        LC_num_classes=17,
        static_dim=3,  # 静态特征数量
        dynamic_dim=17,  # 动态特征数量
        embed_dim=128,
        num_heads=2,
        num_layers=4,
        seq_len=1  # 确保与预测时的 seq_len 一致
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model = accelerator.prepare(model)  # 先 prepare 模型
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # 设置 strict=False
    
    model.to(device)
    model.eval()
    return model


def predict(csv_path, model, device, dynamic_scaler, static_scaler, scaler3ls,seq_len=1, batch_size=32):
    # 读取CSV
    data = pd.read_csv(csv_path)
    data['year'] = data['year'].astype(str)
    data['y'] = data['year'].str.split('_').str.get(0).astype(int)
    # data['m'] = data['year'].str.split('_').str.get(1).astype(int)

    # 假设CSV包含以下列，请根据实际情况修改
    feature_cols = [
    'DEM', 'E', 'Eb', 'Ei', 'Ep', 'Et','p',
     'Ew', 'H', 'LST', 'NDVI', 'rf','S','LAI',"EVI",
     'SMrz','SMs', 'tmp', 'aspect','slope' ,'LC',
]

    static_feature_cols = ['DEM', 'aspect', 'slope']  # 静态特征

    dataLC = data['LC']
    dynamic_feature_cols = [col for col in feature_cols if col not in static_feature_cols + ['LC']]  # 动态特征（不包括 LC 列）

    lat_col = 'lat'  # 纬度列名
    lon_col = 'lon'  # 经度列名

    target_col = 'GWS'  # 目标列名（如果有）

    # 准备数据

    # 使用 transform 而不是 fit_transform
    dynamic_features_scaled = pd.DataFrame(dynamic_scaler.transform(data[dynamic_feature_cols]), columns=dynamic_feature_cols)
    dynamic_features_scaled['LC'] = dataLC
    dynamic_features = dynamic_features_scaled
    year = data['y']
    month = data['y']
    lat = data[lat_col]
    lon = data[lon_col]
    static_features_scaled = pd.DataFrame(static_scaler.transform(data[static_feature_cols]), columns=static_feature_cols)
    year =scaler3ls.transform(year.values.reshape(-1, 1)).reshape(-1)
    year=  pd.DataFrame(year)
    # 由于是预测，没有目标变量，可以填充任意值或重复最后一个目标
    ls = np.zeros(len(data))
    # dictls ={}
    # dictls[target_col]=ls
    
    # targets = pd.DataFrame(dictls)
    targets = ls
    

    # 创建数据集
    dataset = GeoSpatialDataset(dynamic_features, lat, lon, static_features_scaled, targets,year,month, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []

    # Initialize tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Predicting", unit="batch")

    with torch.no_grad():
        for dynamic_x, lat_batch, lon_batch, static_features_batch,year_batch,month_batch,_ in progress_bar:
            dynamic_x = dynamic_x.to(device)
            lat_batch = lat_batch.to(device)
            lon_batch = lon_batch.to(device)
            static_features_batch = static_features_batch.to(device)
            year_batch = year_batch.to(device)
            month_batch = month_batch.to(device)

            outputs = model(dynamic_x, lat_batch, lon_batch, static_features_batch,year_batch,month_batch)
           
            preds = outputs.cpu().numpy().flatten()
            all_preds.extend(preds)

            # 更新进度条
            progress_bar.set_postfix({"Predictions": len(all_preds)})

    # 对齐预测结果与原始数据，通过补偿序列长度
    # prediction_series = pd.Series([None] * seq_len + list(all_preds), name='pre')
    prediction_series = pd.Series(list(all_preds), name='pre')

    # 将预测结果添加到原始数据中
    data_with_preds = pd.concat([data, prediction_series], axis=1)

    return data_with_preds


def mainpred(input_csv,output_csv):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型路径
    model_path = './checkpoints/best_model_iter_72000_rmse_3.4156.pth'


    # 加载模型
    model = load_model(model_path, device)
    # print(model)

    # 加载缩放器
    dynamic_scaler, static_scaler ,scaler3ls= load_scalers()

    # 进行预测
    data_with_preds = predict(input_csv, model, device, dynamic_scaler, static_scaler,scaler3ls, seq_len=1, batch_size=1024+64)

    # 保存结果
    data_with_preds.to_csv(output_csv, index=False)
    print(f"预测完成，结果已保存到 {output_csv}")
    return output_csv




def csv_trans(input_csv,year,outputcsv):
    # 读取CSV文件到DataFrame中
    df =dd.read_csv(input_csv)
    df = df.compute() 
    # print(df)
    # 过滤出需要的年份
    # df = df[(df['year'] >= 2002) & (df['year'] <=2023)]
    df = df[df['year'] == year]

    # pivot操作，使用grid_code作为索引，以year为时间，组合新的列名
    pivot_df = df.pivot_table(index=['grid_code', 'OID_', 'pointid', 'DEM', 'slope', 'aspect', 'lat', 'lon',],
                            columns='year',
                            values=['GWS',"pre"])

    # 展开多层索引的列名
    pivot_df.columns = [f"{var}_{year}" for var, year in pivot_df.columns]

    # 重置索引
    pivot_df.reset_index(inplace=True)

    # 保存新表格到CSV文件
    pivot_df.to_csv(outputcsv, index=False)
    # './Data/Chian_clean_modified_001_DL_pre_tran_1024.csv'



def mian (input_csv_or,outputcsv_or,year,outputcsv_trans):
    output_csvpred = mainpred(input_csv_or,outputcsv_or)
    csv_trans(output_csvpred,year,outputcsv_trans)
    


if __name__ == "__main__":
    for i in range(2003,2024,1):
        year =str(i)
        mian(f'./Data/Data001/Chian_clean_modified_001_{year}.csv',f'./Data/Data001/Chian_clean_modified_001_DL_pre_{year}.csv',i,f'./Data/Data001/Trans_clean_modified_001_DL_{year}.csv')