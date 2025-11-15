# -*- coding: utf-8 -*-
"""
① 从附件1拆分 A/B/C 三类订单/供货数据
② 计算供货次数、平均供货量
"""
import numpy as np
import pandas as pd
import math

# ---------- 1. 读附件1 ----------
file = '../resource/附件1 近5年402家供应商的相关数据.xlsx'
order = pd.read_excel(file, sheet_name=0)   # 订货
supply = pd.read_excel(file, sheet_name=1)  # 供货

# ---------- 2. 数据预处理：剔除极低质量供货商 ----------
print('\n=== 数据预处理：检测极低质量供货商 ===')

# 统计每个供应商的缺货情况
low_quality_suppliers = []

for idx in range(len(supply)):
    supplier_id = supply.iloc[idx]['供应商ID']
    supply_data = supply.iloc[idx].drop(['材料分类', '供应商ID']).values
    order_data = order.iloc[idx].drop(['材料分类', '供应商ID']).values
    
    # 计算差值
    diff = np.abs(supply_data - order_data)
    
    # 统计
    supply_count = (supply_data > 0).sum()  # 供货次数
    diff_gt_100 = (diff > 100).sum()  # 差值>100的次数
    diff_gt_1000 = (diff > 1000).sum()  # 差值>1000的次数
    
    # 判断标准：供货次数少且差值>100次数多
    # 根据题目分析：供货次数<70 且 差值>100次数>=6（刚好33家）
    if supply_count < 70 and diff_gt_100 >= 6:
        low_quality_suppliers.append({
            '供应商ID': supplier_id,
            '供货次数': supply_count,
            '缺货>100': diff_gt_100,
            '缺货>1000': diff_gt_1000
        })

print(f'检测到 {len(low_quality_suppliers)} 家极低质量供货商')

# 保存剔除的供应商列表
if len(low_quality_suppliers) > 0:
    df_low_quality = pd.DataFrame(low_quality_suppliers)
    df_low_quality.to_excel('removed_suppliers.xlsx', index=False)
    print(f'剔除供应商详情已保存到 removed_suppliers.xlsx')
    
    # 从原始数据中剔除这些供应商
    low_quality_ids = [s['供应商ID'] for s in low_quality_suppliers]
    order = order[~order['供应商ID'].isin(low_quality_ids)].reset_index(drop=True)
    supply = supply[~supply['供应商ID'].isin(low_quality_ids)].reset_index(drop=True)
    print(f'剔除后剩余供应商: {len(supply)} 家')

# ---------- 3. 按材料分类拆表 ----------
print('\n=== 按材料分类拆分数据 ===')
cate_map = {'A': 'A', 'B': 'B', 'C': 'C'}
for cate in cate_map:
    # 订货
    tmp_o = order[order['材料分类'] == cate].reset_index(drop=True)
    tmp_o.to_excel(f'order_{cate}.xlsx', index=False)
    # 供货
    tmp_s = supply[supply['材料分类'] == cate].reset_index(drop=True)
    tmp_s.to_excel(f'supply_{cate}.xlsx', index=False)
    print(f'{cate}类: {len(tmp_s)} 家供应商')

# ---------- 4. 计算6个指标 ----------
print('\n=== 计算供应商评价指标 ===')

def calc_six_indicators(cate: str):
    """
    计算6个评价指标（按图片定义）：
    1. 供货次数 - 5年内供货周数总和
    2. 平均供货量 - (总供货量/p_i) ÷ 供货次数
    3. 单次最大供货量 - 历史最大一次供货量/p_i
    4. 供货稳定性 - 供货量与订单量的均方误差MSE（越小越稳定）
    5. 供货连续性 - 基于间隔次数、平均间隔周数、平均连续供货周数综合评分
    6. 合理供货比例 - 在订单±20%范围内的次数比例
    """
    # 读取供货和订货数据
    s = pd.read_excel(f'supply_{cate}.xlsx')
    o = pd.read_excel(f'order_{cate}.xlsx')
    
    # 材料转换系数 p_i（生产1立方米产品所需原材料）
    p_map = {'A': 0.6, 'B': 0.66, 'C': 0.72}
    p_i = p_map[cate]
    
    # 去掉非数字列
    s_data = s.drop(columns=['材料分类', '供应商ID'], errors='ignore')
    o_data = o.drop(columns=['材料分类', '供应商ID'], errors='ignore')
    
    results = []
    for i in range(len(s_data)):
        supply = s_data.iloc[i].values  # 供货数据
        order = o_data.iloc[i].values   # 订货数据
        
        # 1. 供货次数（5年内供货周数总和）
        supply_count = (supply > 0).sum()
        
        # 2. 平均供货量（换算为可生产产品数量后的平均值）
        # m_i = (1/n_i) * Σ(x_{i,t} / p_i)
        supply_converted = supply / p_i  # 转换为可生产产品数
        total_supply_converted = supply_converted.sum()
        avg_supply = total_supply_converted / supply_count if supply_count > 0 else 0
        
        # 3. 单次最大供货量（换算后）
        max_supply = supply_converted.max() if len(supply) > 0 else 0
        
        # 4. 供货稳定性（均方误差MSE）
        # δ_i = (1/n_i) * Σ(x_{i,t} - z_{i,t})^2
        # 只计算有供货的周次
        mask = supply > 0
        if mask.sum() > 0:
            mse = np.mean((supply[mask] - order[mask]) ** 2)
        else:
            mse = 0
        
        # 5. 供货连续性（三个子指标）
        # 子指标1: 间隔次数（断供次数，成本型）
        interruptions = 0
        intervals = []  # 记录断供间隔长度
        consecutive_periods = []  # 记录每段连续供货的长度
        current_gap = 0
        current_consecutive = 0
        in_supply = False
        
        for val in supply:
            if val > 0:
                if in_supply and current_gap > 0:
                    interruptions += 1
                    intervals.append(current_gap)
                in_supply = True
                current_gap = 0
                current_consecutive += 1
            else:
                if in_supply and current_consecutive > 0:
                    consecutive_periods.append(current_consecutive)
                    current_consecutive = 0
                if in_supply:
                    current_gap += 1
        
        # 最后一段连续供货
        if current_consecutive > 0:
            consecutive_periods.append(current_consecutive)
        
        # 子指标2: 平均间隔周数（成本型，越少越好）
        avg_interval = np.mean(intervals) if len(intervals) > 0 else 0
        
        # 子指标3: 平均连续供货周数（效益型，越多越好）
        avg_consecutive = np.mean(consecutive_periods) if len(consecutive_periods) > 0 else 0
        
        # 暂存三个子指标，稍后用熵权法合成
        continuity_sub = {
            'interruptions': interruptions,
            'avg_interval': avg_interval,
            'avg_consecutive': avg_consecutive
        }
        
        # 6. 合理供货比例（在订单±20%范围内）
        reasonable_count = 0
        valid_weeks = 0
        for s_val, o_val in zip(supply, order):
            if o_val > 0:  # 只看有订单的周
                valid_weeks += 1
                if o_val * 0.8 <= s_val <= o_val * 1.2:
                    reasonable_count += 1
        reasonable_ratio = reasonable_count / valid_weeks if valid_weeks > 0 else 0
        
        results.append({
            '供货次数': supply_count,
            '平均供货量': avg_supply,
            '单次最大供货量': max_supply,
            '供货稳定性MSE': mse,
            '间隔次数': continuity_sub['interruptions'],
            '平均间隔周数': continuity_sub['avg_interval'],
            '平均连续周数': continuity_sub['avg_consecutive'],
            '合理供货比例': reasonable_ratio
        })
    
    df_result = pd.DataFrame(results)
    
    # 用熵权法计算连续性三个子指标的权重并合成
    continuity_data = df_result[['间隔次数', '平均间隔周数', '平均连续周数']].copy()
    
    # 成本型指标归一化: x_Scale = (x_max - x) / (x_max - x_min)
    # 效益型指标归一化: x_Scale = (x - x_min) / (x_max - x_min)
    
    continuity_normalized = continuity_data.copy()
    
    # 间隔次数 - 成本型
    col = '间隔次数'
    x_max, x_min = continuity_data[col].max(), continuity_data[col].min()
    if x_max > x_min:
        continuity_normalized[col] = (x_max - continuity_data[col]) / (x_max - x_min)
    else:
        continuity_normalized[col] = 1.0
    
    # 平均间隔周数 - 成本型
    col = '平均间隔周数'
    x_max, x_min = continuity_data[col].max(), continuity_data[col].min()
    if x_max > x_min:
        continuity_normalized[col] = (x_max - continuity_data[col]) / (x_max - x_min)
    else:
        continuity_normalized[col] = 1.0
    
    # 平均连续周数 - 效益型
    col = '平均连续周数'
    x_max, x_min = continuity_data[col].max(), continuity_data[col].min()
    if x_max > x_min:
        continuity_normalized[col] = (continuity_data[col] - x_min) / (x_max - x_min)
    else:
        continuity_normalized[col] = 1.0
    
    # 计算连续性子指标的熵权
    def entropy_single(vec):
        """计算单列指标的熵值"""
        vec = vec[vec > 0]
        if len(vec) == 0:
            return 1.0
        p = vec / vec.sum()
        return -np.sum(p * np.log(p)) / math.log(len(vec))
    
    beta = []
    for col in continuity_normalized.columns:
        t = continuity_normalized[col].values
        t = t / (t.sum() + 1e-10)  # 比重化
        beta.append(entropy_single(t))
    beta = np.array(beta)
    continuity_weights = (1 - beta) / ((1 - beta).sum() + 1e-10)
    
    # 加权合成连续性综合得分
    df_result['供货连续性'] = (continuity_normalized.values * continuity_weights).sum(axis=1)
    
    # 对稳定性MSE取倒数（转换为效益型指标）
    df_result['供货稳定性'] = 1 / (df_result['供货稳定性MSE'] + 1)
    
    # 删除中间的子指标列，只保留最终的6个指标
    df_result = df_result.drop(columns=['供货稳定性MSE', '间隔次数', '平均间隔周数', '平均连续周数'])
    
    # 打印连续性子指标权重
    print(f'\n{cate}类材料 - 连续性子指标权重:')
    for col, w in zip(['间隔次数', '平均间隔周数', '平均连续周数'], continuity_weights):
        print(f'  {col}: {w:.4f} ({w*100:.2f}%)')
    
    return df_result

# 计算三类材料的指标
df_a = calc_six_indicators('A')
df_b = calc_six_indicators('B')
df_c = calc_six_indicators('C')

# 合并所有材料（假设A/B/C各有若干家供应商）
df_all = pd.concat([df_a, df_b, df_c], ignore_index=True)

# 数据质量筛选：只去掉从未供货的供应商
df_filtered = df_all[df_all['供货次数'] > 0]

print(f'\n=== 最终结果 ===')
print(f'预处理后供应商数: {len(df_all)}')
print(f'筛选后供应商数: {len(df_filtered)}')

# 保存表格2
df_filtered.to_excel('table2.xlsx', index=False)
print('已生成 table2.xlsx')