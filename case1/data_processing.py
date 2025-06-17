import pandas as pd
import numpy as np


def clean_and_convert_data(file_path):
    # 定义列名
    column_names = [
        'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
        'trip_distance', 'pickup_longitude', 'pickup_latitude', 'RatecodeID',
        'store_and_fwd_flag', 'dropoff_longitude', 'dropoff_latitude', 'payment_type',
        'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'improvement_surcharge', 'total_amount'
    ]

    # 读取CSV文件
    data = pd.read_csv(file_path, header=None, names=column_names)

    # 将日期时间字段转换为datetime类型
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'], errors='coerce')
    data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'], errors='coerce')

    # 将数值字段转换为float类型
    numeric_columns = ['passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude', 'fare_amount', 'extra', 'mta_tax',
                       'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 查看数据的基本信息
    print("数据集的基本信息:")
    print(data.info())

    # 查看数据的前几行
    print("\n数据预览:")
    print(data.head())

    # 检查缺失值
    print("\n缺失值统计:")
    print(data.isnull().sum())

    return data


def main():
    file_path = '/tmp/gather.csv'
    cleaned_data = clean_and_convert_data(file_path)

    # 如果需要，可以将处理后的数据保存到新的CSV文件
    output_path = '/tmp/processed_gather.csv'
    cleaned_data.to_csv(output_path, index=False)
    print(f"\n处理后的数据已保存到: {output_path}")


if __name__ == "__main__":
    main()
