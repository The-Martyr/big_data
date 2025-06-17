import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
font_path = '/usr/share/fonts/cjkuni-ukai/ukai.ttc'  # AR PL UKai 字体路径
if os.path.exists(font_path):
    font_prop = FontProperties(fname=font_path, size=12)
else:
    print("AR PL UKai 字体文件未找到，请检查字体路径。")
    font_prop = None

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 合并后的信贷数据路径
merged_file_path = "merged_financial_data.csv"

# 读取合并后的信贷数据
merged_df = pd.read_csv(merged_file_path)

# 筛选用于建模的评估数据（数据来源标识为1）
modeling_data = merged_df[merged_df['数据来源标识'] == 1].copy()

# 定义特征和目标变量
features = ['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数']
X = modeling_data[features]
y = modeling_data['审核标识']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
models = {
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '支持向量机': SVC(probability=True, kernel='linear', random_state=42),
    '朴素贝叶斯': GaussianNB(),
    '梯度提升树': GradientBoostingClassifier(random_state=42),
}

# 训练模型并计算ROC曲线数据
roc_data = {}
for model_name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    except Exception as e:
        print(f"训练{model_name}模型时出错: {e}")

# 绘制ROC曲线图
plt.figure(figsize=(10, 8))

for model_name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], label=f'{model_name} (AUC = {data["auc"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')

# 设置图表的标题、轴标签和刻度标签字体
plt.xlabel('假阳性率', fontproperties=font_prop)
plt.ylabel('真阳性率', fontproperties=font_prop)
plt.title('模型ROC曲线比较', fontproperties=font_prop)
plt.legend(loc='lower right', prop=font_prop)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图表到当前目录
plt.savefig('model_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("五个模型的ROC曲线图已保存为 'model_roc_curves.png'")