# Import system modules
import arcpy
from arcpy import sa
import os
# Set property to overwrite existing outputs
# arcpy.env.overwriteOutput = True

# # Set the work space to a gdb
# arcpy.env.workspace = r"./arcgisproProject.gdb"

# Forest-based model taking advantage of both distance features and 
# explanatory rasters. The training and prediction data has been manually
# split so the percentage to exclude parameter was set to 0. A variable importance
# table is created to help assess results and advanced options have been used
# to fine tune the model.

# 剔除离群点函数
def remove_outliers(raster, threshold=5):
    mean_value = float(arcpy.GetRasterProperties_management(raster, "MEAN").getOutput(0))
    std_dev = float(arcpy.GetRasterProperties_management(raster, "STD").getOutput(0))
    lower_bound = mean_value - threshold * std_dev
    upper_bound = mean_value + threshold * std_dev
    
    # 将离群值设置为 NoData
    print(raster)
    filtered_raster = sa.Con((raster >= lower_bound) & (raster <= upper_bound), raster, sa.SetNull(raster, raster))
    return filtered_raster

# IDW插值函数
def idw_interpolation(raster, cell_size=10):
    # 将栅格转换为点，忽略 NoData
    in_memory_points = "in_memory/points"
    arcpy.RasterToPoint_conversion(raster, in_memory_points, "VALUE")
    
    # 执行IDW插值
    idw_result = sa.Idw(in_memory_points, "grid_code", cell_size)
    
    return idw_result

# Set property to overwrite existing outputs
arcpy.env.overwriteOutput = True

# Set the work space to a gdb
arcpy.env.workspace = r"./GWSA计算.gdb"

ls = "Geomamba"
for i in range(2003,2024):
    print(i)


    arcpy.management.XYTableToPoint(f"D:\ARCGISPRO\GWSA计算\DataGeoMamba\Trans_clean_modified_001_DL_{str(i)}.csv",
                                    f"D:/ARCGISPRO/GWSA计算/tmp.gdb/Trans_clean_modified_001_DL_{str(i)}_point",
                                    "lon", "lat", "DEM", 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],VERTCS["WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PARAMETER["Vertical_Shift",0.0],PARAMETER["Direction",1.0],UNIT["Meter",1.0]];-400 -400 1000000000;-100000 10000;-100000 10000;8.98315284119521E-09;0.001;0.001;IsHighPrecision')
    arcpy.conversion.FeatureToRaster(f"D:/ARCGISPRO/GWSA计算/tmp.gdb/Trans_clean_modified_001_DL_{str(i)}_point",
                                     f"GWS_{str(i)}", 
                                     f"D:/ARCGISPRO/GWSA计算/tmp.gdb/GWS_{ls}_001_{str(i)}_tmp",
                                     0.01)

    # raster_current = f"D:/ARCGISPRO/GWSA计算/tmp.gdb/GWS_{ls}_001_{str(i)}_tmp"
    # # 读取DL
    # current_raster = sa.Raster(raster_current)
    #   # 剔除离群点（阈值设为5）
    # filtered = remove_outliers(current_raster, threshold=5)
    # output_path = os.path.join(arcpy.env.workspace, f"GWS_001_{str(i)}_Filtered")
    # filtered.save(output_path)
        
    # # 执行IDW插值，填补NoData区域
    # interpolated_raster = idw_interpolation(filtered, cell_size=0.01)
        
    #     # 合并剔除后的栅格和插值结果，填补NoData区域
    # final_change_rate = sa.Con(sa.IsNull(filtered), interpolated_raster, filtered)
    # output_path = os.path.join(arcpy.env.workspace, f"GWS_001_{str(i)}_{ls}_Filtered_IDW")
    # final_change_rate.save(output_path)
    # out_raster = arcpy.sa.ExtractByMask(output_path, "西北内陆河三级流域", "INSIDE", '73.561173394 35.236540392 106.958946228 49.176265739067 GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],VERTCS["WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PARAMETER["Vertical_Shift",0.0],PARAMETER["Direction",1.0],UNIT["Meter",1.0]]');
    # output_path2 = os.path.join(arcpy.env.workspace, f"GWS_001_{str(i)}_{ls}_NLH_mask")
    # out_raster.save(output_path2)


# arcpy.ddd.Idw("Chian_clean_modified_005_DL_pre_tran_1024", "pre_2002", r"H:/ARC_GIS Pro/MyProject24/arcgisproProject.gdb/DL_Idw10_2002", 0.01, 2, "VARIABLE 12", None)

# for i in range(2002,2024):
#     year = str(i)
#     print(year)
#     arcpy.ddd.Idw("Chian_clean_modified_005_pre_ML_transTable", f"gwspre_{year}", f"H:/ARC_GIS Pro/MyProject24/arcgisproProject.gdb/ML_Idw10_{year}", 0.01, 2, "VARIABLE 12", None)
#     out_raster = arcpy.sa.ExtractByMask( f"H:/ARC_GIS Pro/MyProject24/arcgisproProject.gdb/ML_Idw10_{year}", "中华人民共和国", "INSIDE", '73.561173394 18.2865403920001 134.771173394 53.5465403920001 GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],VERTCS["WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PARAMETER["Vertical_Shift",0.0],PARAMETER["Direction",1.0],UNIT["Meter",1.0]]'); out_raster.save( f"H:/ARC_GIS Pro/MyProject24/arcgisproProject.gdb/ML_Idw10_{year}_Mask")
#     # arcpy.conversion.FeatureToRaster(f"DL_005", f"pre_{year}", f"H:/ARC_GIS Pro/MyProject24/arcgisproProject.gdb/DL_{year}", 0.05)

# prediction_type = "PREDICT_FEATURES"
# in_features = r"point_Project_Clip"
# variable_predict = "GWS_2022"
# output_features = r"GWS_2020"
# treat_variable_as_categorical = None

# ls=['E','Eb','Ei','Ep','Es','Et','etp',"EVI","Ew",'H','LST','NDVI','rf','S','SMrz','SMs','tmp','TWS']
# ls2 = ['CLCD',"LC"]
# static_ls = ["aspect",'DEM','slope1']
# varbible = []
# for i in [2021,2022]:
#     for j in ls:
#         varbible.append([j+"_"+str(i),"false"])
#     for j2 in ls2:
#         varbible.append([j2+"_"+str(i),"true"])

# for i in static_ls:
#     varbible.append([i,"false"])
    

# print(varbible)


# explanatory_variables =varbible
# distance_features = None
# explanatory_rasters = None
# features_to_predict = r"point_Project_Clip"

# output_raster = None
# explanatory_variable_matching = []
# for i in [2019,2020]:
#     for j in ls:
#         explanatory_variable_matching.append([j+"_"+str(i),"false"])
#     for j2 in ls2:
#         explanatory_variable_matching.append([j2+"_"+str(i),"true"])

# for i in static_ls:
#     explanatory_variable_matching.append([i,"false"])
    
    
# explanatory_distance_matching = None
# explanatory_rasters_matching =None
# output_trained_features = r"Training_Output"
# output_importance_table = r"Variable_Importance"
# use_raster_values = False
# number_of_trees = 500
# minimum_leaf_size = 5
# maximum_level = 100
# sample_size = 100



# arcpy.stats.Forest(prediction_type, in_features, variable_predict,
#     treat_variable_as_categorical, explanatory_variables, distance_features,
#     explanatory_rasters, features_to_predict, output_features, output_raster,
#     explanatory_variable_matching, explanatory_distance_matching, 
#     explanatory_rasters_matching, output_trained_features, output_importance_table,
#     use_raster_values, number_of_trees, minimum_leaf_size, maximum_level,
#     sample_size)
# (("Fuzzy_DL_2023" - "Fuzzy_DL_2018")/ "Fuzzy_DL_2018"+ ("Fuzzy_DL_2018" - "Fuzzy_DL_2013")/ "Fuzzy_DL_2013"+ ("Fuzzy_DL_2013" - "Fuzzy_DL_2008")/ "Fuzzy_DL_2008"+ ("Fuzzy_DL_2008" - "Fuzzy_DL_2003")/ "Fuzzy_DL_2003")/4


