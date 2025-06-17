# -*- coding: utf-8 -*-
import pandas as pd

def merge_data(cleaned_apply_file, result_file, output_file):
    column_names = [
        '申请时间', '用户ID', '认证年', '认证月', '认证日', '身份证是否存在',
        '是否认证', '是否电话认证', '认证间隔时间', '银行卡数量', '储蓄卡比例',
        '信用卡比例', '性别', '年龄', '订单总额', '均值', '标准差',
        '预留不同电话数量', '贷款次数', '审核标识'
    ]
    apply_df = pd.read_csv(cleaned_apply_file)
    apply_df.columns = apply_df.columns.str.strip() 
    apply_df['数据来源标识'] = '申请'

    result_df = pd.read_csv(result_file, names=column_names, header=None, low_memory=False)
    result_df.columns = result_df.columns.str.strip()
    result_df['数据来源标识'] = '评估'

    for col in column_names:
        if col not in apply_df.columns:
            apply_df[col] = None  

    apply_df = apply_df[result_df.columns]
    numeric_cols = ['认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例',
                    '年龄', '订单总额', '均值', '标准差',
                    '预留不同电话数量', '贷款次数']
    for col in numeric_cols:
        apply_df[col] = pd.to_numeric(apply_df[col], errors='coerce')
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    for df in [apply_df, result_df]:
        df['性别'] = df['性别'].astype(str).str.strip().str.lower()

    merged_df = pd.concat([apply_df, result_df], ignore_index=True)
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✅ Merged data saved to {}".format(output_file))

if __name__ == '__main__':
    merge_data('cleaned_apply.csv', 'risk_rate.csv', 'merged.csv')