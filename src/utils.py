import platform
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import cross_validate
from .config import logger


def setup_chinese_font():
    """
    根据操作系统自动配置 matplotlib 中文字体。
    """
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(y_true, y_pred, y_prob, model_name="Model"):
    """
    输出分类报告并绘制混淆矩阵、ROC 曲线和 PR 曲线。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率
        model_name: 模型名称
    """
    setup_chinese_font()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"--- {model_name} 评估报告 ---")
    logger.info(f"{'='*50}")
    
    # 生成精确率、召回率、F1分数
    report = classification_report(y_true, y_pred, target_names=['Phishing (0)', 'Legitimate (1)'])
    logger.info(f"\n{report}")

    # 计算AUC得分
    auc = roc_auc_score(y_true, y_prob)
    logger.info(f"ROC-AUC 得分: {auc:.4f}")
    
    # 计算平均精确度
    avg_precision = average_precision_score(y_true, y_prob)
    logger.info(f"Average Precision 得分: {avg_precision:.4f}")

    # 绘制三个子图：混淆矩阵、ROC曲线、PR曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['预测-钓鱼(0)', '预测-合法(1)'],
                yticklabels=['实际-钓鱼(0)', '实际-合法(1)'])
    axes[0].set_title(f'混淆矩阵：{model_name}')
    axes[0].set_ylabel('实际标签')
    axes[0].set_xlabel('预测标签')

    # 2. ROC 曲线
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[1], 
                                     plot_chance_level=True,
                                     name=f"ROC (AUC={auc:.4f})")
    axes[1].set_title(f'ROC 曲线：{model_name}')

    # 3. PR 曲线
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[2],
                                            plot_chance_level=True,
                                            name=f"PR (AP={avg_precision:.4f})")
    axes[2].set_title(f'PR 曲线：{model_name}')
    
    plt.tight_layout()
    plt.show()


def cross_validate_model(model, X, y, cv=5):
    """
    执行 K-Fold 交叉验证并输出详细指标。
    
    Args:
        model: sklearn 兼容的模型或 Pipeline
        X: 特征数据
        y: 标签数据
        cv: 交叉验证折数
        
    Returns:
        cv_results: 交叉验证结果字典
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"--- 执行 {cv}-Fold 交叉验证 ---")
    logger.info(f"{'='*50}")
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        logger.info(f"{metric.upper():>12}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results


def get_error_analysis(X_test, y_test, y_pred, original_df) -> Tuple:
    """
    提取预测错误的原始样本进行追溯。
    
    Args:
        X_test: 测试集特征
        y_test: 测试集真实标签
        y_pred: 测试集预测标签
        original_df: 原始数据框
        
    Returns:
        fn_df: 漏报样本（实际为钓鱼，预测为合法）
        fp_df: 误报样本（实际为合法，预测为钓鱼）
    """
    # 漏报：实际为0（钓鱼），预测为1（合法）
    fn_indices = X_test[(y_test == 0) & (y_pred == 1)].index
    # 误报：实际为1（合法），预测为0（钓鱼）
    fp_indices = X_test[(y_test == 1) & (y_pred == 0)].index

    return original_df.loc[fn_indices], original_df.loc[fp_indices]