import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(row):
    # 地球半径（公里）
    R = 6371.0
    
    # 获取经纬度
    lat1 = row['接单纬度']
    lon1 = row['接单经度']
    lat2 = row['目的地纬度']
    lon2 = row['目的地经度']
    
    # 转换为弧度
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # 计算经度和纬度的差值
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    
    # Haversine 公式
    a = sin(delta_lat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # 计算距离（公里）
    distance = R * c
    return distance

def calculate_distance(input_file, output_file):
    # 读取清洗后的数据
    data = pd.read_csv(input_file)
    
    # 计算每行的距离
    data['distance'] = data.apply(haversine_distance, axis=1)
    
    # 选择需要的列
    output_data = data[['平台标识', '日期', 'distance', '总计费用']]
    
    # 重命名列名
    output_data.columns = ['VendorID', 'date', 'distance', 'total_amount']
    
    # 保存到新的CSV文件
    output_data.to_csv(output_file, index=False)
    
    print(f"数据处理完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    input_file = '/tmp/clear.csv'
    output_file = '/tmp/distance.csv'
    calculate_distance(input_file, output_file)
