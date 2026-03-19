import os

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODEL_SAVE_PATH


class NaiveBayesModel:
    # 使用 __slots__ 限制属性，减少大规模实验时的内存开销
    __slots__ = ['model', '_is_fitted']

    def __init__(self, use_scaler=True, var_smoothing=1e-9):
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

    def train(self, X_train, y_train):
        """执行模型训练"""
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X_test):
        """执行类别预测"""
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先执行 train()")
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """获取预测概率，用于计算AUC"""
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练")
            # 直接获取正类概率
        return self.model.predict_proba(X_test)[:, 1]

    def save_model(self, filename="phish_nb.pkl"):
        """将训练好的模型持久化到本地"""
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        save_path = os.path.join(MODEL_SAVE_PATH, filename)

        # compress=3 可以显著减小 .pkl 文件体积，且加载速度几乎不受影响
        joblib.dump(self.model, save_path, compress=3)
        print(f"[系统] 训练成果（含预处理）已保存: {save_path}")


    @classmethod
    def load_model(cls, filename="phish_nb_v2.pkl"):
        """快速加载已存在的实验流水线"""
        load_path = os.path.join(MODEL_SAVE_PATH, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"未找到模型文件: {load_path}")

        instance = cls()
        instance.model = joblib.load(load_path)
        instance._is_fitted = True
        return instance