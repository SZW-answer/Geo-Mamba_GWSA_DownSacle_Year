import pandas as pd
import re
from tqdm import tqdm

# 输入和输出文件路径
input_csv = ("I:\csv数据处理\csv数据处理\全球025_2.csv")
output_csv = '转换.csv'

# 定义基础变量和按年份分类的变量

# 定义基础变量和按年份分类的变量
base_vars = ['lon', 'lat', 'aspect', 'DEM', 'grid_code',"OID_","pointid","slope"]
yeared_vars = [
    'E', 'Eb', 'Ec', 'Ei', 'Ep', 'Es', 'Et',
    'EV', 'Ew', 'GA', 'H', 'LAI', 'LT','ND','FR',
    'NI', 'pn', 'S', 'Sz', 'Ss', 'tp',"p", "LC",
]

# 提取所有年份
pattern = re.compile(r'_(\d{4})_(\d{2})$') # 匹配以 _年份 结尾的列名

# 读取列名
df_cols = pd.read_csv(input_csv, nrows=0)

years_and_months = set()
for col in df_cols.columns:
    match = pattern.search(col)
    if match:
        year = match.group(1)
        month = match.group(2)
        years_and_months.add((year+"_"+month))

# 将结果排序
years = sorted(years_and_months)

# 准备输出文件的列
output_columns = [
    'grid_code', 'OID_', 'pointid', 'year',
    'lon', 'lat', 'aspect', 'DEM', 'slope'
] + yeared_vars

# 初始化输出 CSV 文件，写入表头
with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
    f_out.write(','.join(output_columns) + '\n')

# 计算总行数用于进度条
with open(input_csv, 'r', encoding='utf-8') as f:
    total_rows = sum(1 for _ in f) - 1  # 减去表头行

# 定义分块大小
chunk_size = 10000# 根据内存情况调整

# 读取 CSV 文件的分块生成器
reader = pd.read_csv(input_csv, chunksize=chunk_size)

# 处理每个块并写入输出文件
with open(output_csv, 'a', newline='', encoding='utf-8') as f_out:
    for chunk in tqdm(reader, total=(total_rows // chunk_size) + 1, desc="Processing chunks"):
        rows = []
        for _, row in chunk.iterrows():
            for year in years:
                new_row = {
                    'grid_code': row['grid_code'],
                    'OID_': row['OID_'],
                    'pointid': row['pointid'],
                    'year': int(year),
                    'lon': row['lon'],
                    'lat': row['lat'],
                    'aspect': row['aspect'],
                    'DEM': row['DEM'],
                    'slope': row['slope']
                }
                for var in yeared_vars:
                    col_name = f"{var}_{year}"
                    new_row[var] = row.get(col_name, None)
                rows.append(new_row)
        # 转换为 DataFrame 并写入 CSV（不写入表头）
        df_out = pd.DataFrame(rows, columns=output_columns)
        df_out.to_csv(f_out, header=False, index=False)

print("数据转换完成，已保存为 转换.csv")
