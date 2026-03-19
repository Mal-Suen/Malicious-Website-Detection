import sys
import os

# 将项目根目录添加到系统路径，确保模块导入正常
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_and_split_data
from src.models.naive_bayes import NaiveBayesModel
from src.utils import evaluate_model, get_error_analysis


def main():
    # 1. 数据预处理
    X_train, X_test, y_train, y_test, raw_df = load_and_split_data()

    # 2. 模型初始化与训练
    nb_experiment = NaiveBayesModel()
    nb_experiment.train(X_train, y_train)

    # 3. 执行预测
    y_pred = nb_experiment.predict(X_test)
    y_prob = nb_experiment.predict_proba(X_test)

    # 4. 评估结果
    evaluate_model(y_test, y_pred, y_prob, model_name="Gaussian Naive Bayes V1.0")

    # 5. 错误追溯分析
    fn_df, fp_df = get_error_analysis(X_test, y_test, y_pred, raw_df)
    print(f"\n[分析] 漏报样本数: {len(fn_df)} | 误报样本数: {len(fp_df)}")

    # 6. 保存模型
    nb_experiment.save_model("phish_nb_v1.pkl")
    print("\n 实验结束，模型已保存。")


if __name__ == "__main__":
    main()