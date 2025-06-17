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

# 筛选数据来源标识为2且审核标识_rf为0的风险用户
risk_users = rf_df[(rf_df['数据来源标识'] == 2) & (rf_df['审核标识_rf'] == 0)]

# 绘制风险用户年龄分布直方图
plt.figure(figsize=(10, 6))
plt.hist(risk_users['年龄'], bins=20, color='#34A853', edgecolor='black')
plt.title('风险用户年龄分布', fontproperties=font_prop)
plt.xlabel('年龄', fontproperties=font_prop)
plt.ylabel('用户数量', fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('risk_user_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制风险用户银行卡数量分布
plt.figure(figsize=(10, 6))
plt.hist(risk_users['银行卡数量'], bins=10, color='#34A853', edgecolor='black')
plt.title('风险用户银行卡数量分布', fontproperties=font_prop)
plt.xlabel('银行卡数量', fontproperties=font_prop)
plt.ylabel('用户数量', fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('risk_user_bank_cards.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制风险用户订单总额分布
plt.figure(figsize=(10, 6))
plt.hist(risk_users['订单总额'], bins=20, color='#34A853', edgecolor='black')
plt.title('风险用户订单总额分布', fontproperties=font_prop)
plt.xlabel('订单总额', fontproperties=font_prop)
plt.ylabel('用户数量', fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('risk_user_order_amount.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制风险用户贷款次数分布
plt.figure(figsize=(10, 6))
plt.hist(risk_users['贷款次数'], bins=10, color='#34A853', edgecolor='black')
plt.title('风险用户贷款次数分布', fontproperties=font_prop)
plt.xlabel('贷款次数', fontproperties=font_prop)
plt.ylabel('用户数量', fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('risk_user_loan_times.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"风险用户年龄分布直方图已保存为 'risk_user_age_distribution.png'")
print(f"风险用户银行卡数量分布直方图已保存为 'risk_user_bank_cards.png'")
print(f"风险用户订单总额分布直方图已保存为 'risk_user_order_amount.png'")
print(f"风险用户贷款次数分布直方图已保存为 'risk_user_loan_times.png'")
