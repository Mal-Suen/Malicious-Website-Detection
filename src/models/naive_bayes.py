import os

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODEL_SAVE_PATH


class NaiveBayesModel:
    def __init__(self, use_scaler=True):
        # 初始化高斯朴素贝叶斯分类器
        # 将 Pipeline 逻辑封装在类内部
        steps = []
        if use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('nb_classifier', GaussianNB()))

        self.model = Pipeline(steps)

    def train(self, X_train, y_train):
        """执行模型训练"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """执行类别预测"""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """获取预测概率，用于计算AUC"""
        return self.model.predict_proba(X_test)[:, 1]

    def save_model(self, filename="phish_nb.pkl"):
        """将训练好的模型持久化到本地"""
        # 统一的保存接口
        save_path = os.path.join(MODEL_SAVE_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"\n实验结束，包含预处理逻辑的完整流水线已保存至: {save_path}")
