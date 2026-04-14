import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE, 
    METADATA_COLS, LABEL_COL, REQUIRED_COLUMNS, ALLOWED_LABELS
)
from .config import logger


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    验证数据框的完整性和有效性。
    
    Args:
        df: 待验证的数据框
        
    Raises:
        FileNotFoundError: 数据文件不存在
        ValueError: 列缺失、标签不合法或存在缺失值
    """
    # 检查必要列是否存在
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {missing_cols}")
    
    # 检查标签列是否只有 0 和 1
    unique_labels = set(df[LABEL_COL].dropna().unique())
    if not unique_labels.issubset(ALLOWED_LABELS):
        raise ValueError(f"标签列包含非法值: {unique_labels - ALLOWED_LABELS}")
    
    # 检查标签列是否有缺失值
    if df[LABEL_COL].isna().any():
        nan_count = df[LABEL_COL].isna().sum()
        raise ValueError(f"标签列存在 {nan_count} 个缺失值")
    
    # 检查特征列缺失值并记录警告
    feature_cols = [col for col in df.columns if col not in METADATA_COLS and col != LABEL_COL]
    nan_features = df[feature_cols].isna().sum()
    nan_features = nan_features[nan_features > 0]
    if not nan_features.empty:
        logger.warning(f"以下特征列存在缺失值，将使用中位数填充:\n{nan_features.to_string()}")


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    填充数值列的缺失值。
    
    Args:
        df: 待处理的数据框
        
    Returns:
        填充后的数据框
    """
    df_clean = df.copy()
    feature_cols = [col for col in df.columns if col not in METADATA_COLS and col != LABEL_COL]
    
    # 数值列使用中位数填充
    for col in feature_cols:
        if df_clean[col].dtype in ['float64', 'int64']:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    # 元数据列使用空字符串填充
    for col in METADATA_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('')
    
    return df_clean


def get_raw_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    仅负责基础的数据加载、验证、填充和分层拆分，保持数据的原始性。
    预处理步骤将由后续的 Pipeline 统一管理。
    
    Returns:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
        df: 原始数据框（用于错误分析）
        
    Raises:
        FileNotFoundError: 数据文件不存在
        ValueError: 数据验证失败
    """
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")
    
    # 读取原始数据
    logger.info(f"正在加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"数据加载完成，共 {len(df)} 条记录")
    
    # 数据验证
    validate_dataframe(df)
    
    # 填充缺失值
    df = fill_missing_values(df)
    
    # 特征选择：剔除非数值文本列和标签列
    X = df.drop(columns=METADATA_COLS + [LABEL_COL])
    y = df[LABEL_COL]
    
    # 数据拆分
    # stratify=y 确保训练集和测试集中0和1的比例与原始数据一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"数据拆分完成: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条")
    return X_train, X_test, y_train, y_test, df