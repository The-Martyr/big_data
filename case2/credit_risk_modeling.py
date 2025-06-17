# # 导入所需的库
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # 连接到华为云服务器并读取数据的路径
# merged_file_path = "merged_financial_data.csv"  # 合并后的信贷数据

# # 读取合并后的信贷数据
# merged_df = pd.read_csv(merged_file_path)

# # 筛选用于建模的评估数据（数据来源标识为1）
# modeling_data = merged_df[merged_df['数据来源标识'] == 1]

# # 定义特征和目标变量
# X = modeling_data[['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']]
# y = modeling_data['审核标识']  # 审核标识作为目标变量，1表示无风险，0表示风险

# # 将性别转换为数值类型（如果性别是分类变量，例如 '男' 和 '女'）
# # 如果性别已经是以数值形式表示（例如 0 和 1），则无需此步骤
# # X['性别'] = X['性别'].map({'男': 0, '女': 1})

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 初始化随机森林分类器
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # 训练随机森林模型
# rf_classifier.fit(X_train, y_train)

# # 预测测试集结果
# y_pred_rf = rf_classifier.predict(X_test)

# # 计算准确率
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"随机森林模型准确率: {accuracy_rf:.2f}")

# # 初始化逻辑回归分类器
# lr_classifier = LogisticRegression(random_state=42, max_iter=1000)

# # 训练逻辑回归模型
# lr_classifier.fit(X_train, y_train)

# # 预测测试集结果
# y_pred_lr = lr_classifier.predict(X_test)

# # 计算准确率
# accuracy_lr = accuracy_score(y_test, y_pred_lr)
# print(f"逻辑回归模型准确率: {accuracy_lr:.2f}")

# # 获取特征重要性
# feature_importances_rf = rf_classifier.feature_importances_
# # 对于逻辑回归，使用系数作为特征重要性
# feature_importances_lr = lr_classifier.coef_[0]

# # 创建特征重要程度数据框
# feature_names = X.columns.tolist()
# importances_rf = pd.Series(feature_importances_rf, index=feature_names)
# importances_lr = pd.Series(feature_importances_lr, index=feature_names)

# # 输出特征重要程度
# print("随机森林特征重要程度:")
# print(importances_rf.sort_values(ascending=False))
# print("\n逻辑回归特征系数:")
# print(importances_lr.sort_values(ascending=False))

# # 保存特征重要程度到CSV
# importances_rf.to_csv("feature_importances_rf.csv", index=True, header=['重要程度系数'])
# importances_lr.to_csv("feature_importances_lr.csv", index=True, header=['系数'])

# # 对信贷申请数据进行预测
# apply_data = merged_df[merged_df['数据来源标识'] == 2]  # 筛选申请数据
# apply_X = apply_data[['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']]

# # 确保性别字段是数值型（如果需要）
# # apply_X['性别'] = apply_X['性别'].map({'男': 0, '女': 1})

# # 使用随机森林模型进行预测
# apply_predictions_rf = rf_classifier.predict(apply_X)

# # 使用逻辑回归模型进行预测
# apply_predictions_lr = lr_classifier.predict(apply_X)

# # 创建预测结果数据框
# apply_data['预测分类_rf'] = apply_predictions_rf
# apply_data['预测分类_lr'] = apply_predictions_lr

# # 保存预测结果到CSV
# apply_data[['用户ID', '预测分类_rf']].to_csv("predictions_rf.csv", index=False)
# apply_data[['用户ID', '预测分类_lr']].to_csv("predictions_lr.csv", index=False)

# print("特征重要程度和预测结果已保存到相应的CSV文件中。")
# 导入所需的库
# 导入所需的库
# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings

# 禁用 SettingWithCopyWarning 警告
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# 连接到华为云服务器并读取数据的路径
merged_file_path = "merged_financial_data.csv"  # 合并后的信贷数据

# 读取合并后的信贷数据
merged_df = pd.read_csv(merged_file_path)

# 筛选用于建模的评估数据（数据来源标识为1）
modeling_data = merged_df[merged_df['数据来源标识'] == 1].copy()  # 使用.copy()确保数据独立

# 定义特征和目标变量
X = modeling_data[['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']]
y = modeling_data['审核标识']  # 审核标识作为目标变量，1表示无风险，0表示风险

# 将性别转换为数值类型（如果性别是分类变量，例如 '男' 和 '女'）
# 如果性别已经是以数值形式表示（例如 0 和 1），则无需此步骤
# X['性别'] = X['性别'].map({'男': 0, '女': 1})

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练随机森林模型
rf_classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred_rf = rf_classifier.predict(X_test)
y_prob_rf = rf_classifier.predict_proba(X_test)[:, 1]  # 用于计算AUC

# 计算评估指标
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"随机森林模型准确率: {accuracy_rf:.2f}")
print(f"随机森林模型精确率: {precision_rf:.2f}")
print(f"随机森林模型召回率: {recall_rf:.2f}")
print(f"随机森林模型AUC: {auc_rf:.2f}")

# 初始化逻辑回归分类器
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)

# 训练逻辑回归模型
lr_classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred_lr = lr_classifier.predict(X_test)
y_prob_lr = lr_classifier.predict_proba(X_test)[:, 1]  # 用于计算AUC

# 计算评估指标
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

print(f"逻辑回归模型准确率: {accuracy_lr:.2f}")
print(f"逻辑回归模型精确率: {precision_lr:.2f}")
print(f"逻辑回归模型召回率: {recall_lr:.2f}")
print(f"逻辑回归模型AUC: {auc_lr:.2f}")

# 获取特征重要性
feature_importances_rf = rf_classifier.feature_importances_
feature_importances_lr = lr_classifier.coef_[0]

# 创建特征重要程度数据框
feature_names = ['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']
importances_rf = pd.DataFrame({'特征': feature_names, '重要程度系数': feature_importances_rf})
importances_lr = pd.DataFrame({'特征': feature_names, '系数': feature_importances_lr})

# 保存特征重要程度到CSV
importances_rf.to_csv("feature_importances_rf.csv", index=False)
importances_lr.to_csv("feature_importances_lr.csv", index=False)

# 对信贷申请数据进行预测
apply_data = merged_df[merged_df['数据来源标识'] == 2].copy()  # 筛选申请数据并创建副本
apply_X = apply_data[['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']]

# 确保性别字段是数值型（如果需要）
# apply_X['性别'] = apply_X['性别'].map({'男': 0, '女': 1})

# 使用随机森林模型进行预测
apply_predictions_rf = rf_classifier.predict(apply_X)
apply_data.loc[:, '预测分类_rf'] = apply_predictions_rf  # 使用.loc进行赋值

# 使用逻辑回归模型进行预测
apply_predictions_lr = lr_classifier.predict(apply_X)
apply_data.loc[:, '预测分类_lr'] = apply_predictions_lr  # 使用.loc进行赋值

# 将预测结果写回合并数据集 - 随机森林
rf_result = merged_df.copy()
rf_result.loc[apply_data.index, '审核标识_rf'] = apply_predictions_rf
rf_result.to_csv("merged_financial_data_rf_predictions.csv", index=False)

# 将预测结果写回合并数据集 - 逻辑回归
lr_result = merged_df.copy()
lr_result.loc[apply_data.index, '审核标识_lr'] = apply_predictions_lr
lr_result.to_csv("merged_financial_data_lr_predictions.csv", index=False)

# 保存预测结果到CSV - 随机森林
predictions_rf = pd.DataFrame({'用户ID': apply_data['用户ID'], '预测分类': apply_predictions_rf})
predictions_rf.to_csv("predictions_rf.csv", index=False)

# 保存预测结果到CSV - 逻辑回归
predictions_lr = pd.DataFrame({'用户ID': apply_data['用户ID'], '预测分类': apply_predictions_lr})
predictions_lr.to_csv("predictions_lr.csv", index=False)

# 统计每个模型预测结果中“数据来源标识”为2且审核标识为0的记录数
rf_risk_count = len(rf_result[(rf_result['数据来源标识'] == 2) & (rf_result['审核标识_rf'] == 0)])
lr_risk_count = len(lr_result[(lr_result['数据来源标识'] == 2) & (lr_result['审核标识_lr'] == 0)])

print(f"随机森林模型预测结果中，数据来源标识为2且审核标识为0的记录数: {rf_risk_count}")
print(f"逻辑回归模型预测结果中，数据来源标识为2且审核标识为0的记录数: {lr_risk_count}")

print("特征重要程度和预测结果已保存到相应的CSV文件中。")