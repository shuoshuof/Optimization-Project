# -*- coding: utf-8 -*-
"""
熵权法通用函数
"""
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing

def entropy(vec: np.ndarray) -> float:
    """计算单列指标的熵值"""
    vec = vec[vec > 0]          # 去掉 0
    if len(vec) == 0:
        return 1.0
    return -np.sum(vec * np.log(vec)) / math.log(len(vec))

def entropy_weight(data: pd.DataFrame) -> np.ndarray:
    """
    对 data(DataFrame) 逐列做 Min-Max 归一化后算熵权
    返回权重向量 ndarray，顺序与列顺序一致
    """
    beta = []
    for col in data.columns:
        t = data[[col]].values
        t = preprocessing.MinMaxScaler().fit_transform(t).ravel()
        t = t / t.sum()        # 比重化
        beta.append(entropy(t))
    beta = np.array(beta)
    return (1 - beta) / (1 - beta).sum()
    # -*- coding: utf-8 -*-
"""
TOPSIS 综合评价
依赖上一步 entropy_weight 函数
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

def topsis(data: pd.DataFrame, weight: np.ndarray = None):
    """
    data: DataFrame，列顺序须为
          [供货次数, 平均供货量, 单次最大供货量, 供货稳定性, 供货连续性, 合理供货比例]
    weight: 外部权重，None 则用熵权法计算
    返回 (Result, Z, weight)
    """
    # 1. 归一化
    t = data.values
    t = preprocessing.MinMaxScaler().fit_transform(t)
    data_norm = pd.DataFrame(t, columns=data.columns)

    # 2. 最优/最劣方案
    z_pos = data_norm.max()
    z_neg = data_norm.min()

    # 3. 权重
    if weight is None:
        weight = entropy_weight(data)

    # 4. 距离
    d_pos = np.sqrt(((data_norm - z_pos) ** 2 * weight).sum(axis=1))
    d_neg = np.sqrt(((data_norm - z_neg) ** 2 * weight).sum(axis=1))

    # 5. 综合得分
    score = d_neg / (d_pos + d_neg + 1e-12)
    data['综合得分指数'] = score
    data['排序'] = data['综合得分指数'].rank(ascending=False)

    return data, pd.DataFrame([z_neg, z_pos], index=['负理想解', '正理想解']), weight

# ---------- 主程序 ----------
if __name__ == '__main__':
    # 读入已合并的 6 指标表
    df = pd.read_excel('table2.xlsx')
    print(f'Total suppliers: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')
    
    Result, Z, weight = topsis(df)
    Result.to_excel('topsis_result.xlsx', index=False)
    
    print('\n权重分配:')
    for col, w in zip(df.columns, weight):
        print(f'  {col}: {w:.4f} ({w*100:.2f}%)')
    print(f'\nResult saved to topsis_result.xlsx')