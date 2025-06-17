import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

# 字体配置
font_path = '/usr/share/fonts/cjkuni-ukai/ukai.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]

# 创建图片目录
picture_dir = 'picture'
if not os.path.exists(picture_dir):
    os.makedirs(picture_dir)

# 读取数据
df = pd.read_csv('/tmp/count.csv')
df['日期'] = df['日期'].astype(str)  # 确保日期为字符串

# 定义颜色方案
bar_colors = ['#4E79A7', '#F28E2B', '#59A14F']  # 蓝/橙/绿
line_colors = ['#E15759', '#76B7B2', '#EDC948']  # 红/青/黄
pie_colors = ['#4E79A7', '#F28E2B']  # 滴滴/Uber颜色

# ================= 上下拼接柱状图 =================
def plot_stacked_bar():
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    formatter = ticker.FuncFormatter(lambda x, _: f'{x/10000:.1f}万')
    
    for idx, platform in enumerate([1, 2]):
        platform_name = '滴滴' if platform == 1 else 'Uber'
        data = df[df['平台标识'] == platform]
        dates = data['日期']
        x = np.arange(len(dates))
        bar_width = 0.25
        
        # 绘制三个指标柱状图
        bars = []
        for i, col in enumerate(['接单总量', '行驶总距离', '车费总收入']):
            values = data[col]
            bar = axs[idx].bar(x + i*bar_width, values, 
                              width=bar_width, 
                              color=bar_colors[i],
                              label=col)
            bars.append(bar)
            
            # 添加数据标签
            for xi, val in zip(x + i*bar_width, values):
                axs[idx].text(xi, val + max(values)*0.02,  # 留2%间距
                             f'{val/10000:.1f}万',
                             ha='center', 
                             va='bottom',
                             fontsize=8)

        # 坐标轴设置
        axs[idx].set_title(f'{platform_name} 基本情况', fontsize=14)
        axs[idx].set_ylabel('数值（单位：万）', fontsize=12)
        axs[idx].yaxis.set_major_formatter(formatter)
        axs[idx].set_xticks(x + bar_width)
        axs[idx].set_xticklabels(dates, rotation=45)
        axs[idx].grid(axis='y', alpha=0.4)
        axs[idx].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, '上下拼接柱状图.png'), dpi=300)
    plt.close()

# ================= 上下拼接折线图 =================
def plot_stacked_line():
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    formatter = ticker.FuncFormatter(lambda x, _: f'{x/10000:.1f}万')
    
    for idx, platform in enumerate([1, 2]):
        platform_name = '滴滴' if platform == 1 else 'Uber'
        data = df[df['平台标识'] == platform]
        dates = data['日期']
        
        # 绘制三条折线
        for i, col in enumerate(['接单总量', '行驶总距离', '车费总收入']):
            values = data[col]
            line = axs[idx].plot(dates, values, 
                               marker='o',
                               color=line_colors[i],
                               linewidth=2,
                               label=col)
            
            # 添加数据标签
            for date, val in zip(dates, values):
                axs[idx].text(date, val, 
                             f'{val/10000:.1f}万',
                             ha='center', 
                             va='bottom',
                             fontsize=8)

        # 坐标轴设置
        axs[idx].set_title(f'{platform_name} 每日走势', fontsize=14)
        axs[idx].set_ylabel('数值（单位：万）', fontsize=12)
        axs[idx].yaxis.set_major_formatter(formatter)
        axs[idx].set_xticks(dates)
        axs[idx].set_xticklabels(dates, rotation=45)
        axs[idx].grid(alpha=0.4)
        axs[idx].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, '上下拼接折线图.png'), dpi=300)
    plt.close()

# ================= 独立饼状图 =================
def plot_pie_charts():
    # 数据筛选（保持原有逻辑）
    start_date = '2016-01-01'
    end_date = '2016-01-06'
    mask = (df['日期'] >= start_date) & (df['日期'] <= end_date)
    df_filtered = df.loc[mask]
    platform_summary = df_filtered.groupby('平台标识')[['接单总量', '行驶总距离', '车费总收入']].sum().reset_index()
    platform_summary['平台名称'] = platform_summary['平台标识'].map({1: '滴滴', 2: 'Uber'})
    
    # 生成三个独立饼图
    metrics = ['接单总量', '行驶总距离', '车费总收入']
    for col in metrics:
        plt.figure(figsize=(8, 8))
        plt.pie(platform_summary[col], 
               labels=platform_summary['平台名称'], 
               autopct='%1.1f%%', 
               startangle=90, 
               colors=pie_colors,
               textprops={'fontsize': 12})
        plt.title(f'{col}比例', fontsize=14)
        plt.savefig(os.path.join(picture_dir, f'{col}比例.png'), dpi=300)
        plt.close()

# ================= 执行主程序 =================
if __name__ == "__main__":
    plot_stacked_bar()    # 生成上下拼接柱状图
    plot_stacked_line()   # 生成上下拼接折线图
    plot_pie_charts()     # 生成独立饼状图
    print(f"图像已保存至：{picture_dir}")