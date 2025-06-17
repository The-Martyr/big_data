import pandas as pd

def clean_data(input_file, output_file):
    # 读取CSV文件
    data = pd.read_csv(input_file)

    # 数据清洗
    # 1. 过滤平台标识不为1或2的记录
    cleaned_data = data[(data['VendorID'] == 1) | (data['VendorID'] == 2)]

    # 2. 过滤接单经度和接单纬度为0的记录
    cleaned_data = cleaned_data[(cleaned_data['pickup_longitude'] != 0) & (cleaned_data['pickup_latitude'] != 0)]

    # 3. 过滤目的地经度和目的地纬度为0的记录
    cleaned_data = cleaned_data[(cleaned_data['dropoff_longitude'] != 0) & (cleaned_data['dropoff_latitude'] != 0)]

    # 4. 过滤其他字段为空的记录
    cleaned_data = cleaned_data.dropna()

    # 5. 只保留日期部分
    cleaned_data['tpep_pickup_datetime'] = pd.to_datetime(cleaned_data['tpep_pickup_datetime']).dt.date

    # 选择需要的列
    cleaned_data = cleaned_data[['VendorID', 'tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'total_amount']]

    # 重命名列名
    cleaned_data.columns = ['平台标识', '日期', '接单经度', '接单纬度', '目的地经度', '目的地纬度', '总计费用']

    # 保存清洗后的数据到新的CSV文件
    cleaned_data.to_csv(output_file, index=False)

    print(f"数据清洗完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    input_file = '/tmp/gather.csv'
    output_file = '/tmp/clear.csv'
    clean_data(input_file, output_file)
