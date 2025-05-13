import os
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

def kalman_smooth(data_column):
    """执行卡尔曼平滑"""
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=data_column.iloc[0],
        initial_state_covariance=1,
        observation_covariance=2.0,  # 观测噪声（根据数据波动性调整）
        transition_covariance=0.75    # 过程噪声（根据系统动态性调整）
    )
    return kf.smooth(data_column.values)[0].flatten()

def process_file(input_path):
    """处理单个文件"""
    df = pd.read_csv(input_path)
    
    # 检查必要列存在
    if not {'time', 'value'}.issubset(df.columns):
        raise ValueError("CSV必须包含'time'和'value'列")
    
    # 排序数据（确保时间序列有序）
    df.sort_values('time', inplace=True)
    
    # 执行滤波
    df['kalman_value'] = kalman_smooth(df['value'])
    
    # 保存结果
    output_path = os.path.splitext(input_path)[0] + '_filtered.csv'
    df[['time', 'value', 'kalman_value']].to_csv(output_path, index=False)
    print(f"已处理: {os.path.basename(output_path)}")

if __name__ == "__main__":
    # 遍历当前目录所有CSV文件（跳过已处理文件）
    for filename in os.listdir('.'):
        if filename.endswith('.csv') and not filename.endswith('_filtered.csv'):
            try:
                process_file(filename)
            except Exception as e:
                print(f"处理 {filename} 失败: {str(e)}")
