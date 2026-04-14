"""
随机森林模型实验脚本。

使用通用实验运行器执行 RF 模型的训练、评估和保存。
"""

import sys
import os

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.random_forest import RandomForestModel
from src.experiment_runner import run_experiment
from src.config import logger


def run_rf_experiment():
    """运行随机森林实验"""
    logger.info("=" * 60)
    logger.info("开始随机森林实验")
    logger.info("=" * 60)
    
    # 初始化模型（可以微调树的数量和深度）
    rf_model = RandomForestModel(n_estimators=100, max_depth=None)
    
    # 运行实验
    results = run_experiment(
        model_instance=rf_model,
        model_name="Random Forest Detector V1.0",
        save_filename="phish_rf_v1.pkl",
        run_cross_validation=True
    )
    
    logger.info("\n实验完成！")
    return results


if __name__ == "__main__":
    try:
        run_rf_experiment()
    except Exception as e:
        logger.error(f"\n[错误] 实验运行失败: {str(e)}", exc_info=True)
        sys.exit(1)
