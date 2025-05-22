

import pandas as pd

# 读取CSV文件到DataFrame中
df = pd.read_csv('./Data/Chian_clean_modified_005_DL_pre.csv')

# 过滤出需要的年份
df = df[(df['year'] >= 2002) & (df['year'] <=2023)]

# pivot操作，使用grid_code作为索引，以year为时间，组合新的列名
pivot_df = df.pivot_table(index=['grid_code', 'OID_', 'pointid', 'DEM', 'slope', 'aspect', 'lat', 'lon',],
                          columns='year',
                          values=['GWS',"pre"])

# 展开多层索引的列名
pivot_df.columns = [f"{var}_{year}" for var, year in pivot_df.columns]

# 重置索引
pivot_df.reset_index(inplace=True)

# 保存新表格到CSV文件
pivot_df.to_csv('./Data/Chian_clean_modified_005_DL_pre_trans.csv', index=False)















# (("Fuzz_ExtractByMaskIDW2023"-"Fuzz_ExtractByMaskIDW2018")/"Fuzz_ExtractByMaskIDW2018"+("Fuzz_ExtractByMaskIDW2018"-"Fuzz_ExtractByMaskIDW2013")/"Fuzz_ExtractByMaskIDW2013"+("Fuzz_ExtractByMaskIDW2013"-"Fuzz_ExtractByMaskIDW2008")/"Fuzz_ExtractByMaskIDW2008"+("Fuzz_ExtractByMaskIDW2008"-"Fuzz_ExtractByMaskIDW2003")/"Fuzz_ExtractByMaskIDW2003")/4