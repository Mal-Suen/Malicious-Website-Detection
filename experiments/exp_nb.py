"""
高斯朴素贝叶斯模型实验脚本。

使用通用实验运行器执行 NB 模型的训练、评估和保存。
"""

import sys
import os

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.naive_bayes import NaiveBayesModel
from src.experiment_runner import run_experiment
from src.config import logger


def run_nb_experiment():
    """运行高斯朴素贝叶斯实验"""
    logger.info("=" * 60)
    logger.info("开始高斯朴素贝叶斯实验")
    logger.info("=" * 60)
    
    # 初始化模型
    nb_model = NaiveBayesModel(use_scaler=True, var_smoothing=1e-9)
    
    # 运行实验
    results = run_experiment(
        model_instance=nb_model,
        model_name="Gaussian Naive Bayes V2.0",
        save_filename="phish_nb_v2.pkl",
        run_cross_validation=True
    )
    
    logger.info("\n实验完成！")
    return results


if __name__ == "__main__":
    try:
        run_nb_experiment()
    except Exception as e:
        logger.error(f"\n[错误] 实验运行失败: {str(e)}", exc_info=True)
        sys.exit(1)
