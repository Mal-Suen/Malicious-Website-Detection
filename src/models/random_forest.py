import os
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE, MODEL_SAVE_PATH, logger


class RandomForestModel:
    """
    随机森林模型类。
    
    采用集成学习的思想，通过构建多棵决策树并进行投票来提高分类精度。
    使用 Pipeline 将标准化和分类器组合在一起。
    """
    # 使用 __slots__ 优化内存，仅允许实例拥有这两个属性
    __slots__ = ['model', '_is_fitted']

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        """
        初始化随机森林流水线。
        
        Args:
            n_estimators: 森林中决策树的数量，默认 100 棵
            max_depth: 每棵树的最大深度，None 表示不限制
        """
        # 构建流水线：标准化 -> 随机森林分类器
        # 即使随机森林对缩放不敏感，保留 StandardScaler 有助于保持特征分布的一致性
        self.model = Pipeline([
            ('scaler', StandardScaler(copy=False)),  # 原地缩放，节省内存
            ('rf_classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=RANDOM_STATE,  # 核心：确保森林生长的过程可复现
                n_jobs=-1
            ))
        ])
        self._is_fitted = False

    def train(self, X_train: Union[pd.DataFrame, np.ndarray], 
              y_train: Union[pd.Series, np.ndarray]) -> None:
        """
        训练模型：同时拟合标准化参数和森林中的决策树。
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Random Forest 模型训练完成")

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        执行类别预测（0 为钓鱼，1 为合法）。
        
        Args:
            X_test: 测试集特征
            
        Returns:
            预测的类别标签
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 train()")
        return self.model.predict(X_test)

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        获取预测概率，用于计算 AUC。
        
        Args:
            X_test: 测试集特征
            
        Returns:
            正类（label=1）的预测概率
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练")
        
        proba_matrix = self.model.predict_proba(X_test)
        # 返回概率矩阵的第二列（即 label=1 的概率）
        return proba_matrix[:, 1]

    def save_model(self, filename: str = "phish_rf_v1.pkl") -> None:
        """
        将完整的流水线（含 Scaler 参数）保存到磁盘。
        
        Args:
            filename: 保存的文件名
        """
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        save_path = os.path.join(MODEL_SAVE_PATH, filename)
        # 使用 compress=3 进行压缩，减小模型占用的磁盘空间
        joblib.dump(self.model, save_path, compress=3)
        logger.info(f"随机森林流水线已保存至: {save_path}")

    @classmethod
    def load_model(cls, filename: str = "phish_rf_v1.pkl") -> 'RandomForestModel':
        """
        类方法：直接从磁盘加载训练好的流水线实例。
        
        Args:
            filename: 模型文件名
            
        Returns:
            加载的模型实例
        """
        load_path = os.path.join(MODEL_SAVE_PATH, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"未找到模型文件: {load_path}")

        instance = cls()
        instance.model = joblib.load(load_path)
        instance._is_fitted = True
        logger.info(f"模型已从 {load_path} 加载")
        return instance