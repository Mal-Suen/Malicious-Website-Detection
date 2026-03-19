import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE, METADATA_COLS, LABEL_COL


def get_raw_splits():
    """
        仅负责基础的数据加载和分层拆分，保持数据的原始性。
        预处理步骤将由后续的 Pipeline 统一管理。
    """
    # 读取原始数据
    df = pd.read_csv(DATA_PATH)

    # 特征选择：剔除非数值文本列和标签列
    # drop函数用于删除指定的列
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

    return X_train, X_test, y_train, y_test, df