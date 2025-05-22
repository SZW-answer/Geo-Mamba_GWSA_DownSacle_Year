# 训练代码部分

import os
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from Datasets import GeoSpatialDataset
from Nets import TransformerRegressor
from utils import evaluate_model, evaluate_model_metrics, train_model_with_evaluation, load_checkpoint
from torch.cuda.amp import GradScaler
# import pygeohash as pgh
import pickle  # 用于保存 scaler 和 feature list
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
torch.cuda.is_available()
torch.multiprocessing.set_sharing_strategy('file_system')
from accelerate import Accelerator  # 导入 Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
from torch.utils.tensorboard import SummaryWriter
# 初始化 TensorBoard
writer = SummaryWriter(log_dir='runs/attention_and_feature_visualization')
# 确保 CUDA 可用
torch.cuda.is_available()

# # 2. 数据预处理
df = dd.read_csv("./Data/Chian_clean_modified_025.csv")

# df = dd.read_csv("./Global025.csv").head(n=50000)

# 定义特征列
# 定义特征列
feature_cols = [
    'DEM', 'E', 'Eb', 'Ei', 'Ep', 'Et','p',
     'Ew', 'H', 'LST', 'NDVI', 'rf','S','LAI',"EVI",
     'SMrz','SMs', 'tmp', 'aspect','slope' ,'LC',
]

# df = df.dropna()
# for col in feature_cols:
#     print(col)
#     df = df[df[col] !=0]

# with ProgressBar():
#     df.to_csv("./Data/Global025_2.csv", index=False, single_file=True)


target_col = 'GWS'

df['year'] = df['year'].astype(str)
df['y'] = df['year'].str.split('_').str.get(0).astype(int)
df['m'] = df['year'].str.split('_').str.get(1).astype(int)

# df['y'] = df['year'].apply(lambda x: int(x.split('_')[0]))
# df['m'] = df['year'].apply(lambda x: int(x.split('_')[1]))



# 定义静态和动态特征
# 定义静态和动态特征
static_cols = ['DEM', 'aspect', 'slope']  # 静态特征
dynamic_cols = [col for col in feature_cols if col not in static_cols + ['LC']]  # 动态特征（不包括 LC 列）


# 分离特征
X = df[feature_cols].compute()
y = df[target_col].compute().values
# 提取纬度和经度
lat = df['lat'].compute()
lon = df['lon'].compute()
year = df['y'].compute()
month =  df['y'].compute()


# # 分离特征
# X = df[feature_cols]
# y = df[target_col].values
# # 提取纬度和经度
# lat = df['lat']
# lon = df['lon']
# year = df['y']
# month = df['m']


# 提取静态和动态特征
static_features = X[static_cols]
dynamic_features = X[dynamic_cols + ['LC']]  # 包括 LC 列

# # 标准化数值特征（不包括 LC 列）
scaler = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
dynamic_features_scaled = pd.DataFrame(scaler.fit_transform(dynamic_features[dynamic_cols]), columns=dynamic_cols)
# dynamic_features_scaled = pd.DataFrame((dynamic_features[dynamic_cols]), columns=dynamic_cols)
# 将标准化后的数据与未标准化的 LC 列合并
dynamic_features_scaled['LC'] = dynamic_features['LC'].values

year =scaler3.fit_transform(year.values.reshape(-1, 1)).reshape(-1)
year=  pd.DataFrame(year)

# 对静态特征进行标准化
static_features = pd.DataFrame(scaler2.fit_transform(static_features), columns=static_cols)

# 标准化目标变量
# scaler_y = StandardScaler()
# y = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

# 保存特征缩放器和特征列表
with open('Deep_scaler_X_d.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('Deep_scaler_X_y.pkl', 'wb') as f:
    pickle.dump(scaler3, f)
# 保存特征列表
with open('Deep_feature_list.pkl', 'wb') as f:
    pickle.dump(dynamic_cols, f)
with open('Deep_scaler_X_s.pkl', 'wb') as f:
    pickle.dump(scaler2, f)



# 划分训练和测试集
# 注意：确保 lat 和 lon 也被划分
X_train_dynamic, X_test_dynamic, y_train, y_test, lat_train, lat_test, lon_train, lon_test, static_train, static_test,year_train,year_test,month_train,month_test= train_test_split(
    dynamic_features_scaled, y, lat, lon, static_features,year,month,
    test_size=0.1, shuffle=True,random_state=32
)
del X,y,dynamic_features_scaled,lat,lon,df,static_features

# 创建数据集和数据加载器
seq_length = 1 # 根据需要调整序列长度
train_dataset = GeoSpatialDataset(
    dynamic_features=X_train_dynamic,
    lat=lat_train,
    lon=lon_train,
    static_features=static_train,
    targets=y_train,
    year= year_train,
    month= month_train,
    seq_len=seq_length
)
test_dataset = GeoSpatialDataset(
    dynamic_features=X_test_dynamic,
    lat=lat_test,
    lon=lon_test,
    static_features=static_test,
    targets=y_test,
    year= year_test,
    month= month_test,
    seq_len=seq_length
)

print("Training set size: ", len(train_dataset))
print("Testing set size: ", len(test_dataset))

# batch_size = 512+512+32+32
batch_size = 64*4

# 根据硬件资源调整批量大小-
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=1)

# 3. 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


num_features_dynamic = len(dynamic_cols)
num_features_static = len(static_cols)
LC_num_classes = 17  # 确保 LC_Type1 的类别数正确

model = TransformerRegressor(
    # 动态特征数量
    LC_num_classes=LC_num_classes,
    static_dim=num_features_static,  # 静态特征数量
    dynamic_dim=num_features_dynamic,  # 动态特征数量
    embed_dim=128,
    num_heads=2,
    num_layers=4,
    seq_len=seq_length
)

# 4. 定义损失函数和优化器
criterion = nn.HuberLoss("mean")
optimizer = optim.AdamW(model.parameters(), lr=3.56e-3,weight_decay=1e-2,eps=1e-8)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, 1500
                                               , eta_min=1e-9, last_epoch=- 1, verbose=True)


# 设置模型保存目录和日志
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

checkpoint_path = os.path.join(save_dir, 'last_model.pth')

# 初始化日志文件
with open(log_file, 'w') as f:
    f.write(f'Training started at {datetime.now()}\n')
    f.write('Epoch\tTrain Loss\tTest Loss\tMSE\tRMSE\tMAE\tR2\tBest R2\n')

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision='no',kwargs_handlers=[kwargs])
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
logger.info("My log", main_process_only=False)
set_seed(42)


# 加载断点（checkpoint）如果存在
model, optimizer, scheduler, start_iter, best_r2 = load_checkpoint(
    model, optimizer, scheduler, checkpoint_path, accelerator, use_amp=(accelerator.mixed_precision != 'no')
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3,weight_decay=1e-2,eps=1e-8)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, 1500
                                               , eta_min=1e-9, last_epoch=- 1, verbose=True)

# 准备模型、优化器、数据加载器和调度器（再次准备，因为重新初始化优化器和调度器）
model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)
use_amp=(accelerator.mixed_precision != 'no')

# print(model)
# 训练模型  
train_model_with_evaluation(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    max_iterations=141980*6,       
    eval_every_iters=1500,       
    save_every_iters=5000,      
    log_file=log_file,      # 日志
    checkpoint_path=checkpoint_path,
    use_amp=(accelerator.mixed_precision != 'no'),
    accelerator=accelerator
)

# 最终评估模型
print("=== 最终模型评估 ===")
final_test_loss = evaluate_model(model, test_loader, criterion)
final_mse, final_rmse, final_mae, final_r2 = evaluate_model_metrics(model, test_loader)

# 记录最终评估结果到日志文件
with open(log_file, 'a') as f:
    f.write(f"Final\t-\t{final_test_loss:.4f}\t{final_mse:.4f}\t{final_rmse:.4f}\t{final_mae:.4f}\t{final_r2:.4f}\t{best_r2:.4f}\n")

print("=== 训练和评估完成 ===")
