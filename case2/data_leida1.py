import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np  # 导入 numpy 库
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
font_path = '/usr/share/fonts/cjkuni-ukai/ukai.ttc'  # AR PL UKai 字体路径
if os.path.exists(font_path):
    font_prop = FontProperties(fname=font_path, size=12)
    plt.rcParams['font.sans-serif'] = ['AR PL UKai']
else:
    print("AR PL UKai 字体文件未找到，请检查字体路径。")

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义文件路径
feature_importances_path = os.path.join('feature_importances_rf.csv')

# 读取特征重要性数据
feature_importances = pd.read_csv(feature_importances_path)

# 提取特征名称和重要性系数
features = feature_importances['特征'].values
importances = feature_importances['重要程度系数'].values

# 创建条形图
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制水平条形图
colors = plt.cm.tab20c(np.linspace(0, 1, len(features)))
bars = ax.barh(features, importances, color=colors)

# 添加数值标签
for bar, importance in zip(bars, importances):
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f"{importance:.3f}",
            ha='left', va='center', fontproperties=font_prop)

# 设置图表标题和轴标签
ax.set_title('风控关键指标重要性（基于随机森林模型）', fontproperties=font_prop)
ax.set_xlabel('重要程度系数', fontproperties=font_prop)
ax.set_ylabel('特征', fontproperties=font_prop)

# 设置y轴标签
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontproperties=font_prop)

# 保存图表到当前目录
plt.savefig('rf_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"随机森林模型的信贷风控关键指标条形图已保存为 'rf_feature_importance_bar.png'")