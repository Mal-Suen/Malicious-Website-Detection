import sys
import os
import joblib  # 用于保存流水线模型
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from src.models.naive_bayes import NaiveBayesModel

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import get_raw_splits
from src.utils import evaluate_model, get_error_analysis
from src.config import MODEL_SAVE_PATH


def main():

    # 获取原始拆分数据
    X_train, X_test, y_train, y_test, raw_df = get_raw_splits()

    # 直接调用封装好的类
    nb_research = NaiveBayesModel(use_scaler=True)

    # 训练与预测
    nb_research.train(X_train, y_train)
    y_pred = nb_research.predict(X_test)
    y_prob = nb_research.predict_proba(X_test)

    # 评估
    evaluate_model(y_test, y_pred, y_prob, model_name="Gaussian Naive Bayes V2.0")

    # 错误追溯分析
    fn_df, fp_df = get_error_analysis(X_test, y_test, y_pred, raw_df)
    print(f"\n[分析] 漏报样本数 (FN): {len(fn_df)} | 误报样本数 (FP): {len(fp_df)}")

    # 保存整个流水线
    nb_research.save_model("phish_nb_v2.pkl")

if __name__ == "__main__":
    main()