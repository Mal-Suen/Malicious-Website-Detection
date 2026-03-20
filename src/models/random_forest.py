import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..config import RANDOM_STATE, MODEL_SAVE_PATH


class RandomForestModel:
    """
    随机森林模型类：
    采用集成学习的思想，通过构建多棵决策树并进行投票来提高分类精度。
    """
    # 使用 __slots__ 优化内存，仅允许实例拥有这两个属性
    __slots__ = ['model', '_is_fitted']

    def __init__(self, n_estimators=100, max_depth=None):
        """
        初始化随机森林流水线。
        :param n_estimators: 森林中决策树的数量，默认 100 棵。
        :param max_depth: 每棵树的最大深度，None 表示不限制。
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

    def train(self, X_train, y_train):
        """
        训练模型：同时拟合标准化参数和森林中的决策树。
        """
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X_test):
        """
        执行类别预测（0 为钓鱼，1 为合法）。
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 train()")
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """获取预测概率，用于计算AUC"""
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练")
        # 返回概率矩阵的第二列（即 label=1 的概率）
        return self.model.predict_proba(X_test)[:, 1]

    def save_model(self, filename="phish_rf_v1.pkl"):
        """
        将完整的流水线（含 Scaler 参数）保存到磁盘。
        """
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        save_path = os.path.join(MODEL_SAVE_PATH, filename)
        # 使用 compress=3 进行压缩，减小模型占用的磁盘空间
        joblib.dump(self.model, save_path, compress=3)
        print(f"\n[系统] 随机森林流水线已保存至: {save_path}")

    @classmethod
    def load_model(cls, filename="phish_rf_v1.pkl"):
        """
        类方法：直接从磁盘加载训练好的流水线实例。
        """
        load_path = os.path.join(MODEL_SAVE_PATH, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"未找到模型文件: {load_path}")

        instance = cls()
        instance.model = joblib.load(load_path)
        instance._is_fitted = True
        return instance