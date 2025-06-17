import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

# 创建雷达图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# 绘制雷达图
theta = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
radii = importances
ax.fill(theta, radii, color='#34A853', alpha=0.25)
ax.plot(theta, radii, color='#34A853', linewidth=2)

# 设置特征标签
ax.set_thetagrids(theta * 180 / np.pi, features, fontproperties=font_prop)

# 添加数值标签
for i in range(len(features)):
    ax.text(theta[i], radii[i] + 0.05, f"{radii[i]:.3f}", ha='center', va='center', fontproperties=font_prop)

# 设置图表标题
ax.set_title('随机森林模型信贷风控关键指标', fontproperties=font_prop)

# 设置雷达图的范围
ax.set_ylim(0, max(importances) * 1.1)

# 添加网格
ax.grid(True)

# 保存图表到当前目录
plt.savefig('rf_feature_importance_radar.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"随机森林模型的信贷风控关键指标雷达图已保存为 'rf_feature_importance_radar.png'")