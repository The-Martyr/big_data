import pandas as pd

def aggregate_gps_data(input_file, output_file):
    # 读取距离计算后的数据
    data = pd.read_csv(input_file)

    # 使用命名聚合并重命名分组键列
    aggregated_data = data.groupby(['VendorID', 'date']).agg(
        行驶总距离=('distance', 'sum'),
        车费总收入=('total_amount', 'sum'),
        接单总量=('VendorID', 'count')
    ).reset_index().rename(columns={'VendorID': '平台标识', 'date': '日期'})

    # 保存到新的CSV文件
    aggregated_data.to_csv(output_file, index=False)

    print(f"数据统计完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    input_file = '/tmp/distance.csv'
    output_file = '/tmp/count.csv'
    aggregate_gps_data(input_file, output_file)
