"""
模型模块。

包含各种机器学习模型的封装类。
"""

from .naive_bayes import NaiveBayesModel
from .random_forest import RandomForestModel

__all__ = [
    'NaiveBayesModel',
    'RandomForestModel',
]
