# 导入所需的库
import pandas as pd

# 连接到华为云服务器并读取数据的路径
apply_file_path = "cleaned_risk_rate_apply.csv"  # 清洗后的信贷申请数据
evaluation_file_path = "risk_rate.csv"  # 信贷评估结果数据

# 读取清洗后的信贷申请数据（已有列名）
apply_df = pd.read_csv(apply_file_path)

# 定义信贷评估结果数据的列名
evaluation_columns = [
    '申请时间', '用户ID', '认证年', '认证月', '认证日', '身份证是否存在', 
    '是否认证', '是否电话认证', '认证间隔时间', '银行卡数量', '储蓄卡比例', 
    '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', 
    '预留不同电话数量', '贷款次数', '审核标识'
]

# 读取信贷评估结果数据（无列名）
evaluation_df = pd.read_csv(evaluation_file_path, header=None)
evaluation_df.columns = evaluation_columns

# 为清洗后的信贷申请数据添加数据来源标识（2表示申请数据）
apply_df['数据来源标识'] = 2

# 为信贷评估结果数据添加数据来源标识（1表示评估数据）
evaluation_df['数据来源标识'] = 1

# 定义合并后需要保留的字段
columns_to_keep = [
    '用户ID', '认证间隔时间', '银行卡数量', '储蓄卡比例', 
    '信用卡比例', '性别', '年龄', '订单总额', '均值', 
    '标准差', '预留不同电话数量', '贷款次数', '数据来源标识'
]

# 对于信贷评估结果数据，还需保留审核标识
evaluation_columns_to_keep = columns_to_keep + ['审核标识']

# 筛选清洗后的信贷申请数据的字段
apply_df = apply_df[columns_to_keep]

# 筛选信贷评估结果数据的字段
evaluation_df = evaluation_df[evaluation_columns_to_keep]

# 合并两份数据
merged_df = pd.concat([evaluation_df, apply_df], ignore_index=True)

# 输出原始数据量和合并后的数据量
print(f"原始申请数据量: {len(apply_df)} 行")
print(f"原始评估数据量: {len(evaluation_df)} 行")
print(f"合并后的数据量: {len(merged_df)} 行")

# 查看合并后的数据概览
print(merged_df.info())
print(merged_df.head())

# 保存合并后的数据
output_file_path = "merged_financial_data.csv"  # 合并后的数据保存路径
merged_df.to_csv(output_file_path, index=False)

print("数据合并完成，已保存到", output_file_path)