"""
恶意网站检测项目核心模块。

包含数据预处理、模型定义和工具函数。
"""

from .config import BASE_DIR, DATA_PATH, MODEL_SAVE_PATH, RANDOM_STATE, TEST_SIZE, CV_FOLDS
from .preprocess import get_raw_splits, validate_dataframe, fill_missing_values
from .utils import evaluate_model, cross_validate_model, get_error_analysis

__all__ = [
    'BASE_DIR',
    'DATA_PATH',
    'MODEL_SAVE_PATH',
    'RANDOM_STATE',
    'TEST_SIZE',
    'CV_FOLDS',
    'get_raw_splits',
    'validate_dataframe',
    'fill_missing_values',
    'evaluate_model',
    'cross_validate_model',
    'get_error_analysis',
]
