# -*- coding: utf-8 -*-
import pandas as pd

def clean_data(apply_file, output_file):
    column_names = [
        '申请时间', '用户ID', '认证年', '认证月', '认证日', '身份证是否存在',
        '是否认证', '是否电话认证', '认证间隔时间', '银行卡数量', '储蓄卡比例',
        '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差',
        '预留不同电话数量', '贷款次数'
    ]
    
    df = pd.read_csv(apply_file, names=column_names, header=None, low_memory=False)
    
    df.columns = df.columns.str.strip()
    print("列名：", df.columns.tolist())
    df = df[df['性别'].isin(['男', '女'])]
    df = df.dropna()

    df.to_csv(output_file, index=False)
    print("✅ Cleaned data saved to {}".format(output_file))

if __name__ == '__main__':
    clean_data('risk_rate_apply.csv', 'cleaned_apply.csv')

