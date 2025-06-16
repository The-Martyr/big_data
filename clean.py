# 导入所需的库
import pandas as pd
import numpy as np

# 连接到华为云服务器并读取数据的路径
file_path = "risk_rate_apply.csv"  # 根据实际路径调整

# 读取数据，此时数据没有字段名
df = pd.read_csv(file_path, header=None)

# 输出原始数据量
original_rows = len(df)
print(f"原始数据量: {original_rows} 行")

# 根据字段描述，为数据集添加字段名
columns = [
    '申请时间', '用户ID', '认证年', '认证月', '认证日', '身份证是否存在', 
    '是否认证', '是否电话认证', '认证间隔时间', '银行卡数量', '储蓄卡比例', 
    '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', 
    '预留不同电话数量', '贷款次数'
]
df.columns = columns

# 去除是否认证状态为未认证的数据（假设未认证状态对应的值是0）
df = df[df['是否认证'] == 1]

# 去除性别不明确的数据（假设性别值应为0或1）
df = df[(df['性别'] == 0) | (df['性别'] == 1)]

# 去除数据为空的字段（包含空值的行）
df = df.dropna(how='any')

# 输出清洗后的数据量
cleaned_rows = len(df)
print(f"清洗后的数据量: {cleaned_rows} 行")



# 保存清洗后的数据
output_file_path = "cleaned_risk_rate_apply.csv"  # 保存路径
df.to_csv(output_file_path, index=False)

print("数据清洗完成，已保存到", output_file_path)
