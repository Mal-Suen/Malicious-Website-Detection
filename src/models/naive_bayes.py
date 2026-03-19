from sklearn.naive_bayes import GaussianNB
import joblib
import os
from ..config import MODEL_SAVE_PATH

class NaiveBayesModel:
    def __init__(self):
        # 初始化高斯朴素贝叶斯分类器
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        """执行模型训练"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """执行类别预测"""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """获取预测概率，用于计算AUC"""
        return self.model.predict_proba(X_test)[:, 1]

    def save_model(self, filename="nb_model.pkl"):
        """将训练好的模型持久化到本地"""
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        joblib.dump(self.model, os.path.join(MODEL_SAVE_PATH, filename))