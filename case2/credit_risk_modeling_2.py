import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 合并后的信贷数据路径
merged_file_path = "merged_financial_data.csv"

# 读取合并后的信贷数据
merged_df = pd.read_csv(merged_file_path)

# 筛选用于建模的评估数据（数据来源标识为1）
modeling_data = merged_df[merged_df['数据来源标识'] == 1].copy()

# 定义特征和目标变量
X = modeling_data[['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']]
y = modeling_data['审核标识']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化梯度提升树分类器
gb_classifier = GradientBoostingClassifier(random_state=42)

# 训练梯度提升树模型
gb_classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred_gb = gb_classifier.predict(X_test)
y_prob_gb = gb_classifier.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_prob_gb)

print(f"梯度提升树模型准确率: {accuracy_gb:.2f}")
print(f"梯度提升树模型精确率: {precision_gb:.2f}")
print(f"梯度提升树模型召回率: {recall_gb:.2f}")
print(f"梯度提升树模型AUC: {auc_gb:.2f}")

# 获取特征重要性
feature_importances_gb = gb_classifier.feature_importances_

# 创建特征重要程度数据框
feature_names = ['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']
importances_gb = pd.DataFrame({'特征': feature_names, '重要程度系数': feature_importances_gb})

# 保存特征重要程度到CSV
importances_gb.to_csv("feature_importances_gb.csv", index=False)

# 对信贷申请数据进行预测
apply_data = merged_df[merged_df['数据来源标识'] == 2].copy()
apply_X = apply_data[['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']]

# 使用梯度提升树模型进行预测
apply_predictions_gb = gb_classifier.predict(apply_X)
apply_data.loc[:, '预测分类_gb'] = apply_predictions_gb

# 将预测结果写回合并数据集
gb_result = merged_df.copy()
gb_result.loc[apply_data.index, '审核标识_gb'] = apply_predictions_gb
gb_result.to_csv("merged_financial_data_gb_predictions.csv", index=False)

# 保存预测结果到CSV
predictions_gb = pd.DataFrame({'用户ID': apply_data['用户ID'], '预测分类': apply_predictions_gb})
predictions_gb.to_csv("predictions_gb.csv", index=False)

# 统计预测结果中“数据来源标识”为2且审核标识为0的记录数
gb_risk_count = len(gb_result[(gb_result['数据来源标识'] == 2) & (gb_result['审核标识_gb'] == 0)])

print(f"梯度提升树模型预测结果中，数据来源标识为2且审核标识为0的记录数: {gb_risk_count}")

print("特征重要程度和预测结果已保存到相应的CSV文件中。")