# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 设置中文字体支持（适配OpenEuler环境）
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'AR PL UKai CN']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False  # 显示负号


def visualize_risk_analysis():
    output_dir = "visualization_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载数据
    df = pd.read_csv('merged.csv', low_memory=False)
    pred_df = pd.read_csv('predictions.csv', low_memory=False)

    # 清洗列名空格
    df.columns = df.columns.str.strip()
    pred_df.columns = pred_df.columns.str.strip()

    # 合并真实标签和预测结果（只保留“评估”数据）
    result_df = df[df['数据来源标识'] == '评估'].copy()
    result_df['预测分类'] = pred_df['预测分类']

    # 1. 风险比例对比图（真实 vs 预测）
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    real_counts = result_df['审核标识'].value_counts(normalize=True) * 100
    sns.barplot(x=real_counts.values, y=real_counts.index, palette="viridis")
    plt.title("真实风险比例（评估数据）", fontsize=14)
    plt.xlabel("百分比 (%)")
    plt.ylabel("审核状态")
    for i, v in enumerate(real_counts.values):
        plt.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)
    plt.subplot(1, 2, 2)
    pred_counts = result_df['预测分类'].value_counts(normalize=True) * 100
    sns.barplot(x=pred_counts.values, y=pred_counts.index, palette="magma")
    plt.title("预测风险比例", fontsize=14)
    plt.xlabel("百分比 (%)")
    plt.ylabel("预测类别")
    for i, v in enumerate(pred_counts.values):
        plt.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_ratio_comparison.png"))
    print("✅ 风险比例图表已保存")

    # 2. 风控关键指标重要性（模型视角）
    model_data = joblib.load('rf_model.pkl')
    model = model_data['model']
    try:
        feature_names = model.feature_names_in_
    except AttributeError:
        feature_names = [
        '认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例',
        '年龄', '订单总额', '均值', '标准差',
        '预留不同电话数量', '贷款次数', '性别_0.0'
        ]

    importances = model.feature_importances_    

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="coolwarm")
    plt.title("风控关键指标重要性（基于模型）", fontsize=16)
    plt.xlabel("重要性分数", fontsize=12)
    plt.ylabel("特征名称", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_sorted.png"))
    print("✅ 特征重要性图表已保存")

    # 3. 关键指标在不同风险类别中的分布对比（直方图 + KDE + 箱线图）
    top_features = importance_df.head(5)['Feature'].tolist()

    for feature in top_features:
        if feature not in result_df.columns or feature.startswith('性别_'):
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})

        # 直方图 + KDE
        sns.histplot(data=result_df, x=feature, hue='审核标识', bins=30, kde=True, ax=axes[0])
        axes[0].set_title(f"【{feature}】在不同风险类别的分布 (Hist + KDE)")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("人数")
        axes[0].legend(title="审核标识", labels=["通过", "未通过"])

        # 箱线图
        sns.boxplot(data=result_df, x='审核标识', y=feature, ax=axes[1], palette="Set2")
        axes[1].set_title(f"【{feature}】箱线图对比")
        axes[1].set_xlabel("审核标识")
        axes[1].set_ylabel(feature)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"distribution_{feature}.png"))
        plt.close()
        print(f"✅ {feature} 分布图已保存")

    print("🎉 所有图表已生成完毕，保存在目录：{}".format(output_dir))


if __name__ == '__main__':
    visualize_risk_analysis()