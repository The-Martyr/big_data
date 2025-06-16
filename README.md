# 📊 金融信贷风险预测项目

本项目基于深度学习模型（Transformer）对用户的信贷申请数据进行风险预测，并输出可视化分析结果。

## 🚀 使用说明

### 1. 安装依赖

```bash
pip install pandas numpy torch rtdl scikit-learn matplotlib seaborn
```

### 2. 准备输入数据

将以下两个 CSV 文件放入项目根目录：

- `risk_rate.csv`：包含已审核用户的信贷数据（用于训练）
- `risk_rate_apply.csv`：包含待预测用户的信贷数据（无审核标识）

> ⚠️ 数据格式为无表头 CSV，字段顺序需与代码中定义一致。

### 3. 运行模型训练与预测

```bash
python clean.py
python merge.py
python run_model.py
```

## 📝 备注

- 所有字段列名已在代码中手动指定，CSV 文件应为无表头格式。
