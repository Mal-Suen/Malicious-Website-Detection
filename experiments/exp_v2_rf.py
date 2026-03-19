import sys
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # 换成随机森林
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import get_raw_splits
from src.utils import evaluate_model, get_error_analysis
from src.config import MODEL_SAVE_PATH


def main():
    X_train, X_test, y_train, y_test, raw_df = get_raw_splits()

    # 构建随机森林流水线
    # 注意：对于随机森林，StandardScaler 其实不是必须的，但放在这里也没坏处
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf_classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("正在通过流水线训练随机森林模型 (V2.0)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 1. 评估
    evaluate_model(y_test, y_pred, y_prob, model_name="随机森林 (Pipeline V2.0)")

    # 2. 错误分析
    fn_df, fp_df = get_error_analysis(X_test, y_test, y_pred, raw_df)
    print(f"\n[分析] 漏报样本数 (FN): {len(fn_df)} | 误报样本数 (FP): {len(fp_df)}")

    # 3. 提取特征重要性 (随机森林特有功能)
    # pipeline.named_steps 可以获取流水线中特定的步骤
    rf_model = pipeline.named_steps['rf_classifier']
    importances = rf_model.feature_importances_
    feat_names = X_train.columns
    print("\n--- 前 5 名核心特征 ---")
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{name}: {imp:.4f}")

    # 4. 保存
    save_path = os.path.join(MODEL_SAVE_PATH, "phish_rf_v2.pkl")
    joblib.dump(pipeline, save_path)
    print(f"\n 随机森林流水线已保存。")


if __name__ == "__main__":
    main()