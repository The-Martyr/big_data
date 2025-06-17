# -*- coding: utf-8 -*-
import pandas as pd
import joblib

def predict(data_file, model_file, output_file):
    # 加载模型包
    model_data = joblib.load(model_file)
    clf = model_data['model']
    feature_names = model_data['feature_names']

    # 读取数据
    df = pd.read_csv(data_file, low_memory=False)
    df.columns = df.columns.str.strip()

    # 特征选择
    features = [
        '认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例',
        '年龄', '订单总额', '均值', '标准差',
        '预留不同电话数量', '贷款次数'
    ]

    X = df[features]

    # 性别清洗 + One-Hot 编码
    if '性别' in df.columns:
        gender_col = df['性别'].astype(str).str.strip().str.lower()
        gender_dummies = pd.get_dummies(gender_col, prefix='性别', drop_first=True)
        X = pd.concat([X.reset_index(drop=True), gender_dummies.reset_index(drop=True)], axis=1)

    # 确保特征顺序一致
    X = X.reindex(columns=feature_names, fill_value=0)

    # 预测
    df['预测分类'] = clf.predict(X)

    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 预测完成，结果已保存至 {output_file}")

if __name__ == '__main__':
    predict('merged.csv', 'rf_model.pkl', 'predictions.csv')