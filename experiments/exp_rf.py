import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocess import get_raw_splits
from src.models.random_forest import RandomForestModel
from src.utils import evaluate_model, get_error_analysis


def run_rf_experiment():
    """
    随机森林实验完整工作流
    """
    # 获取原始拆分数据
    print("\n[处理] 正在加载并拆分数据...")
    X_train, X_test, y_train, y_test, raw_df = get_raw_splits()

    # 初始化并训练模型
    # 注意：RandomForestModel 内部会自动从 config 获取 RANDOM_STATE
    print("\n[处理] 正在初始化并训练模型...")
    rf_research = RandomForestModel(n_estimators=100)  # 可以微调树的数量
    rf_research.train(X_train, y_train)

    # 预测与性能评估
    print("\n[分析] 训练完成！执行测试集评估...")
    y_pred = rf_research.predict(X_test)
    y_prob = rf_research.predict_proba(X_test)

    # 评估
    evaluate_model(y_test, y_pred, y_prob, model_name="Random Forest Detector V1.0")

    # 错误分析
    fn_df, fp_df = get_error_analysis(X_test, y_test, y_pred, raw_df)
    print(f"\n[分析] 漏报样本数 (FN): {len(fn_df)} | 误报样本数 (FP): {len(fp_df)}")

    if len(fp_df) > 0:
        print("\n--- 误报样本 (FP) 原始信息抽样 (前 5 条) ---")
        # 显示 URL 及其关键特征
        print(fp_df[['URL', 'URLSimilarityIndex', 'NoOfSubDomain']].head())
    else:
        print("\n恭喜！随机森林在测试集上实现了 0 误报。")

    # 实验成果持久化
    rf_research.save_model("phish_rf_v1.pkl")
    print("\n实验流水线已保存，可用于后期部署。")


if __name__ == "__main__":
    try:
        run_rf_experiment()
    except Exception as e:
        print(f"\n[错误] 实验运行失败: {str(e)}")
