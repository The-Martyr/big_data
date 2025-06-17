import pandas as pd
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
rf_predictions_path = os.path.join('merged_financial_data_rf_predictions.csv')

# 读取随机森林模型的预测结果数据
rf_df = pd.read_csv(rf_predictions_path)

# 过滤掉年龄中的异常值（例如，年龄小于0或大于100的值）
rf_df_filtered = rf_df[(rf_df['年龄'] >= 10) & (rf_df['年龄'] <= 100)]

# 绘制用户年龄分布直方图
plt.figure(figsize=(10, 6))
plt.hist(rf_df_filtered['年龄'], bins=20, color='#34A853', edgecolor='black')
plt.title('用户年龄分布', fontproperties=font_prop)
plt.xlabel('年龄', fontproperties=font_prop)
plt.ylabel('用户数量', fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('user_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制性别比例饼状图
gender_counts = rf_df['性别'].value_counts()
male_count = gender_counts.get(1, 0)
female_count = gender_counts.get(0, 0)

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# 饼状图
wedges, texts, autotexts = ax.pie(
    [male_count, female_count],
    labels=['男', '女'],
    autopct='%1.2f%%',
    colors=['#4285F4', '#34A853'],
    startangle=90
)

# 设置中文标签字体
for text in texts + autotexts:
    text.set_fontproperties(font_prop)

ax.set_title('性别比例', fontproperties=font_prop)
ax.legend(loc='lower left', prop=font_prop)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 保存图表到当前目录
plt.savefig('gender_ratio_pie.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"用户年龄分布直方图已保存为 'user_age_distribution.png'")
print(f"性别比例饼状图已保存为 'gender_ratio_pie.png'")