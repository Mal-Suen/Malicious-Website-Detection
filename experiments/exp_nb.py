import os
import sys

from src.models.naive_bayes import NaiveBayesModel

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import get_raw_splits
from src.utils import evaluate_model, get_error_analysis


def run_nb_experiment():
    # 获取原始拆分数据
    print("\n[处理] 正在加载并拆分数据...")
    X_train, X_test, y_train, y_test, raw_df = get_raw_splits()

    # 初始化并训练模型
    print("\n[处理] 正在初始化并训练模型...")
    nb_research = NaiveBayesModel(use_scaler=True)
    nb_research.train(X_train, y_train)

    # 预测与性能评估
    print("\n[分析] 训练完成！执行测试集评估...")
    y_pred = nb_research.predict(X_test)
    y_prob = nb_research.predict_proba(X_test)

    # 评估
    evaluate_model(y_test, y_pred, y_prob, model_name="Gaussian Naive Bayes V2.0")

    # 错误分析
    fn_df, fp_df = get_error_analysis(X_test, y_test, y_pred, raw_df)
    print(f"\n[分析] 漏报样本数 (FN): {len(fn_df)} | 误报样本数 (FP): {len(fp_df)}")

    if len(fp_df) > 0:
        print("\n--- 误报样本 (FP) 原始信息抽样 (前 5 条) ---")
        # 显示 URL 及其关键特征
        print(fp_df[['URL', 'URLSimilarityIndex', 'NoOfSubDomain']].head())
    else:
        print("\n恭喜！高斯朴素贝叶斯在测试集上实现了 0 误报。")

    # 实验成果持久化
    nb_research.save_model("phish_nb_v2.pkl")
    print("\n实验流水线已保存，可用于后期部署。")


if __name__ == "__main__":
    try:
        run_nb_experiment()
    except Exception as e:
        print(f"\n[错误] 实验运行失败: {str(e)}")
