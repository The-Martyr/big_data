# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_model(data_file):
    df = pd.read_csv(data_file, low_memory=False)
    df.columns = df.columns.str.strip() 
    df_train = df[df['数据来源标识'] == '评估'].copy()
    if df_train.empty:
        raise ValueError("⚠️ 没有符合条件的评估数据，请检查数据过滤条件")
    features = [
        '认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例',
        '年龄', '订单总额', '均值', '标准差',
        '预留不同电话数量', '贷款次数'
    ]
    X = df_train[features]
    y = df_train['审核标识']
    gender_col = df_train['性别'].astype(str).str.strip().str.lower()
    gender_dummies = pd.get_dummies(gender_col, prefix='性别', drop_first=True)
    X = pd.concat([X.reset_index(drop=True), gender_dummies.reset_index(drop=True)], axis=1)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = GradientBoostingClassifier(
        n_estimators=100,  
        learning_rate=0.1,   
        max_depth=3,          
        random_state=42     
    )
    clf.fit(X_train, y_train)
    model_data = {
        'model': clf,
        'feature_names': X.columns.tolist(),  
        'class_names': clf.classes_.tolist() 
    }
    joblib.dump(model_data, 'gb_model.pkl')
    print("✅ 梯度提升树模型已保存至 gb_model.pkl")
    y_pred = clf.predict(X_test)
    print("📊 分类评估报告（测试集）:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_model('merged.csv')