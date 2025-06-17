import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings

# 禁用 SettingWithCopyWarning 警告
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

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

# 初始化支持向量机分类器，使用线性核
svm_classifier = SVC(kernel='linear', probability=True, random_state=42)  # 显式指定线性核

# 训练支持向量机模型
svm_classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred_svm = svm_classifier.predict(X_test)
y_prob_svm = svm_classifier.predict_proba(X_test)[:, 1]  # 用于计算AUC

# 计算评估指标
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_prob_svm)

print(f"支持向量机模型准确率: {accuracy_svm:.2f}")
print(f"支持向量机模型精确率: {precision_svm:.2f}")
print(f"支持向量机模型召回率: {recall_svm:.2f}")
print(f"支持向量机模型AUC: {auc_svm:.2f}")

# 初始化朴素贝叶斯分类器
nb_classifier = GaussianNB()

# 训练朴素贝叶斯模型
nb_classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred_nb = nb_classifier.predict(X_test)
y_prob_nb = nb_classifier.predict_proba(X_test)[:, 1]  # 用于计算AUC

# 计算评估指标
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
auc_nb = roc_auc_score(y_test, y_prob_nb)

print(f"朴素贝叶斯模型准确率: {accuracy_nb:.2f}")
print(f"朴素贝叶斯模型精确率: {precision_nb:.2f}")
print(f"朴素贝叶斯模型召回率: {recall_nb:.2f}")
print(f"朴素贝叶斯模型AUC: {auc_nb:.2f}")

# 获取特征重要性（SVM 使用线性核时可用）
feature_importances_svm = np.abs(svm_classifier.coef_[0])

# Naive Bayes 没有直接的特征重要性属性，可以使用类条件特征均值或其他统计方法
# 这里为了演示，简单地使用特征在预测中的权重
feature_importances_nb = nb_classifier.theta_[1] - nb_classifier.theta_[0]

# 创建特征重要程度数据框
feature_names = ['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']
importances_svm = pd.DataFrame({'特征': feature_names, '重要程度系数': feature_importances_svm})
importances_nb = pd.DataFrame({'特征': feature_names, '重要程度系数': feature_importances_nb})

# 保存特征重要程度到CSV
importances_svm.to_csv("feature_importances_svm.csv", index=False)
importances_nb.to_csv("feature_importances_nb.csv", index=False)

# 对信贷申请数据进行预测
apply_data = merged_df[merged_df['数据来源标识'] == 2].copy()
apply_X = apply_data[['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']]

# 使用支持向量机模型进行预测
apply_predictions_svm = svm_classifier.predict(apply_X)
apply_data.loc[:, '预测分类_svm'] = apply_predictions_svm

# 使用朴素贝叶斯模型进行预测
apply_predictions_nb = nb_classifier.predict(apply_X)
apply_data.loc[:, '预测分类_nb'] = apply_predictions_nb

# 将预测结果写回合并数据集 - 支持向量机
svm_result = merged_df.copy()
svm_result.loc[apply_data.index, '审核标识_svm'] = apply_predictions_svm
svm_result.to_csv("merged_financial_data_svm_predictions.csv", index=False)

# 将预测结果写回合并数据集 - 朴素贝叶斯
nb_result = merged_df.copy()
nb_result.loc[apply_data.index, '审核标识_nb'] = apply_predictions_nb
nb_result.to_csv("merged_financial_data_nb_predictions.csv", index=False)

# 保存预测结果到CSV
predictions_svm = pd.DataFrame({'用户ID': apply_data['用户ID'], '预测分类': apply_predictions_svm})
predictions_svm.to_csv("predictions_svm.csv", index=False)

predictions_nb = pd.DataFrame({'用户ID': apply_data['用户ID'], '预测分类': apply_predictions_nb})
predictions_nb.to_csv("predictions_nb.csv", index=False)

# 统计每个模型预测结果中“数据来源标识”为2且审核标识为0的记录数
svm_risk_count = len(svm_result[(svm_result['数据来源标识'] == 2) & (svm_result['审核标识_svm'] == 0)])
nb_risk_count = len(nb_result[(nb_result['数据来源标识'] == 2) & (nb_result['审核标识_nb'] == 0)])

print(f"支持向量机模型预测结果中，数据来源标识为2且审核标识为0的记录数: {svm_risk_count}")
print(f"朴素贝叶斯模型预测结果中，数据来源标识为2且审核标识为0的记录数: {nb_risk_count}")

print("特征重要程度和预测结果已保存到相应的CSV文件中。")