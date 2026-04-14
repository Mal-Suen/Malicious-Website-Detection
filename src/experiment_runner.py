"""
通用实验运行器模块。

提供统一的模型实验流程，消除重复代码。
"""

from typing import Any, Dict
from .config import CV_FOLDS, logger
from .preprocess import get_raw_splits
from .utils import evaluate_model, cross_validate_model, get_error_analysis


def run_experiment(
    model_instance: Any,
    model_name: str,
    save_filename: str,
    run_cross_validation: bool = True,
    cv_folds: int = CV_FOLDS
) -> Dict[str, Any]:
    """
    运行完整的模型实验流程，包括：
    - 数据加载和拆分
    - 模型训练
    - 测试集评估
    - 交叉验证（可选）
    - 错误分析
    - 模型保存
    
    Args:
        model_instance: 模型实例（需支持 train, predict, predict_proba, save_model 方法）
        model_name: 模型名称，用于日志和报告
        save_filename: 保存的文件名
        run_cross_validation: 是否运行交叉验证
        cv_folds: 交叉验证折数
        
    Returns:
        results: 包含实验结果的字典
    """
    results = {
        'model_name': model_name,
        'save_filename': save_filename,
    }
    
    # 1. 获取原始拆分数据
    logger.info("\n[处理] 正在加载并拆分数据...")
    X_train, X_test, y_train, y_test, raw_df = get_raw_splits()
    
    # 2. 初始化并训练模型
    logger.info("\n[处理] 正在初始化并训练模型...")
    model_instance.train(X_train, y_train)
    
    # 3. 预测与性能评估
    logger.info("\n[分析] 训练完成！执行测试集评估...")
    y_pred = model_instance.predict(X_test)
    y_prob = model_instance.predict_proba(X_test)
    
    # 4. 评估
    evaluate_model(y_test, y_pred, y_prob, model_name=model_name)
    
    # 5. 交叉验证（可选）
    if run_cross_validation:
        logger.info(f"\n[分析] 执行 {cv_folds}-Fold 交叉验证...")
        cv_results = cross_validate_model(
            model_instance.model, X_train, y_train, cv=cv_folds
        )
        results['cv_results'] = cv_results
    
    # 6. 错误分析
    fn_df, fp_df = get_error_analysis(X_test, y_test, y_pred, raw_df)
    results['false_negatives'] = len(fn_df)
    results['false_positives'] = len(fp_df)
    
    logger.info(f"\n[分析] 漏报样本数 (FN): {len(fn_df)} | 误报样本数 (FP): {len(fp_df)}")
    
    if len(fp_df) > 0:
        logger.info("\n--- 误报样本 (FP) 原始信息抽样 (前 5 条) ---")
        # 尝试显示 URL 及其关键特征
        display_cols = ['URL', 'URLSimilarityIndex', 'NoOfSubDomain']
        available_cols = [col for col in display_cols if col in fp_df.columns]
        if available_cols:
            logger.info(f"\n{fp_df[available_cols].head()}")
    else:
        logger.info(f"\n恭喜！{model_name} 在测试集上实现了 0 误报。")
    
    # 7. 实验成果持久化
    model_instance.save_model(save_filename)
    logger.info("\n实验流水线已保存，可用于后期部署。")
    
    return results
