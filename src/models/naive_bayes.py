import os
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODEL_SAVE_PATH, logger


class NaiveBayesModel:
    """
    高斯朴素贝叶斯模型封装类。
    
    使用 Pipeline 将标准化和分类器组合在一起，确保预处理步骤
    与模型一起保存和加载。
    """
    # 使用 __slots__ 限制属性，减少大规模实验时的内存开销
    __slots__ = ['model', '_is_fitted']

    def __init__(self, use_scaler: bool = True, var_smoothing: float = 1e-9):
        """
        初始化高斯朴素贝叶斯模型。
        
        Args:
            use_scaler: 是否使用 StandardScaler 进行特征标准化
            var_smoothing: 方差平滑参数，用于防止零方差
        """
        # 初始化高斯朴素贝叶斯分类器
        # 将 Pipeline 逻辑封装在类内部
        steps = []
        if use_scaler:
            # 标准化对于 GaussianNB 消除量纲偏差至关重要
            steps.append(('scaler', StandardScaler(copy=False)))  # copy=False 减少内存拷贝

        # 核心算法：var_smoothing 能够平滑特征方差，提升鲁棒性
        steps.append(('nb_classifier', GaussianNB(var_smoothing=var_smoothing)))

        self.model = Pipeline(steps)
        self._is_fitted = False

    def train(self, X_train: Union[pd.DataFrame, np.ndarray], 
              y_train: Union[pd.Series, np.ndarray]) -> None:
        """
        执行模型训练。
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Naive Bayes 模型训练完成")

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        执行类别预测。
        
        Args:
            X_test: 测试集特征
            
        Returns:
            预测的类别标签
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先执行 train()")
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
        # 直接获取正类概率（sklearn 中 classes_ 按升序排列，1 在索引 1）
        return proba_matrix[:, 1]

    def save_model(self, filename: str = "phish_nb_v2.pkl") -> None:
        """
        将训练好的模型持久化到本地。
        
        Args:
            filename: 保存的文件名
        """
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        save_path = os.path.join(MODEL_SAVE_PATH, filename)

        # compress=3 可以显著减小 .pkl 文件体积，且加载速度几乎不受影响
        joblib.dump(self.model, save_path, compress=3)
        logger.info(f"训练成果（含预处理）已保存: {save_path}")

    @classmethod
    def load_model(cls, filename: str = "phish_nb_v2.pkl") -> 'NaiveBayesModel':
        """
        快速加载已存在的实验流水线。
        
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