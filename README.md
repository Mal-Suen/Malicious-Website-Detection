# Malicious-Website-Detection

> **From Basic Classification to Industrial-Grade Detection: Engineering Optimization of Phishing Website Detection Models.**
> **从基础分类到工业级检测：钓鱼网站检测模型的工程化优化与验证。**

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 🇬🇧 English Documentation

### 📖 Introduction

Phishing websites remain one of the most prevalent cyber threats, costing organizations billions annually. While deep learning approaches achieve high accuracy, they suffer from **computational overhead**, **interpretability gaps**, and **deployment complexity**. This project transforms a basic machine learning prototype into a rigorous, production-ready detection system. By employing **Stratified Sampling**, **K-Fold Cross Validation**, and **Strict Data Validation Pipelines**, we demonstrate that properly engineered classical models (Naive Bayes, Random Forest) can achieve **ROC-AUC > 0.95**, offering a robust, explainable, and deployable alternative for real-time phishing detection.

### 🚀 Key Features & Conclusions

| Feature | Detail |
| :--- | :--- |
| **🧪 High Accuracy** | Multi-metric validation: ROC-AUC, PR-AUC, F1-Score. Target: **>0.95 AUC**. |
| **🧹 Strict Preprocessing** | **Automated data validation** must precede model training. Missing values handled via median imputation. |
| **📊 Comprehensive Evaluation** | Beyond accuracy: Confusion Matrix, ROC Curves, PR Curves, Error Analysis (FP/FN). |
| **🏭 Industrial Architecture** | Pipeline-based design: `Validate → Preprocess → Train → Evaluate → Deploy`. |
| **🔄 Cross-Validation** | 5-Fold CV ensures statistical significance, not single-split luck. |
| **⚡ Production-Ready** | Model serialization with compression, ready for API deployment. |

### 📊 Experimental Results

Validated on PhiUSIIL Phishing URL Dataset:

| Configuration | ROC-AUC | F1-Score | Conclusion |
| :--- | :---: | :---: | :--- |
| **Naive Bayes (Baseline)** | ~0.92 | ~0.91 | ✅ Fast, interpretable |
| **Naive Bayes (Optimized)** | **>0.95** | **>0.94** | ✅ With scaling + smoothing |
| **Random Forest (100 trees)** | **>0.98** | **>0.97** | ✅ Best overall performance |
| **5-Fold CV (RF)** | **0.97 ± 0.01** | **0.96 ± 0.01** | ✅ Statistically robust |

### 🛠️ Getting Started

#### Installation

```bash
# Clone repository
git clone https://github.com/Mal-Suen/Malicious-Website-Detection.git
cd Malicious-Website-Detection

# Install dependencies
pip install -r requirements.txt
```

#### Python API

```python
from src.models.random_forest import RandomForestModel
from src.preprocess import get_raw_splits
from src.utils import evaluate_model

# Load data
X_train, X_test, y_train, y_test, raw_df = get_raw_splits()

# Initialize & Train
rf_model = RandomForestModel(n_estimators=100, max_depth=None)
rf_model.train(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_prob, model_name="Random Forest V1.0")

# Save for deployment
rf_model.save_model("phish_rf_v1.pkl")
```

#### CLI Commands

```bash
# Run Naive Bayes experiment
python experiments/exp_nb.py

# Run Random Forest experiment
python experiments/exp_rf.py

# Each experiment outputs:
# - Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC)
# - Visualization (Confusion Matrix, ROC Curve, PR Curve)
# - Error analysis (False Positives/Negatives)
# - Serialized model in saved_models/
```

### 📂 Project Structure

```text
Malicious-Website-Detection/
├── data/                          # Datasets
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
├── experiments/                   # Experiment scripts
│   ├── exp_nb.py                  # Naive Bayes workflow
│   └── exp_rf.py                  # Random Forest workflow
├── saved_models/                  # Serialized models
│   ├── phish_nb_v2.pkl            # Trained NB model
│   └── phish_rf_v1.pkl            # Trained RF model
├── src/                           # Core Python package
│   ├── __init__.py                # Module exports
│   ├── config.py                  # Configuration & constants
│   ├── preprocess.py              # Data loading & validation
│   ├── utils.py                   # Evaluation & visualization
│   ├── experiment_runner.py       # Generic experiment pipeline
│   └── models/                    # Model implementations
│       ├── __init__.py
│       ├── naive_bayes.py         # Gaussian NB classifier
│       └── random_forest.py       # Random Forest classifier
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

### 🔬 Key Findings

1. **Data Validation is Critical:** Unvalidated datasets with missing values can degrade model performance by 5-10%. Automated validation must precede training.
2. **Standardization Matters:** Even for tree-based models, feature scaling improves convergence and maintains consistency across pipeline variants.
3. **Single Split is Insufficient:** One train/test split may reflect random chance. K-Fold cross-validation (K≥5) is required for statistical significance.
4. **Error Analysis Drives Improvement:** Analyzing false positives/negatives reveals feature engineering opportunities (e.g., URL length, domain entropy).

---

## 🇨🇳 中文文档

### 📖 项目简介

钓鱼网站依然是最普遍的网络安全威胁之一，每年给组织造成数十亿美元的损失。虽然深度学习方法能够达到高准确率，但存在**计算开销大**、**可解释性差**和**部署复杂**的问题。本项目将基础的机器学习原型重构为严谨的、可投入生产的检测系统。通过采用**分层抽样**、**K 折交叉验证**和**严格的数据验证流水线**，我们证明了经过工程化优化的经典模型（朴素贝叶斯、随机森林）可以实现 **ROC-AUC > 0.95**，为实时钓鱼检测提供了一种鲁棒、可解释且可部署的替代方案。

### 🚀 核心特性与结论

| 特性 | 细节 |
| :--- | :--- |
| **🧪 高精度** | 多指标验证：ROC-AUC、PR-AUC、F1 分数。目标：**>0.95 AUC**。 |
| **🧹 严格预处理** | **自动数据验证**必须在模型训练之前执行。缺失值使用中位数填充。 |
| **📊 全面评估** | 超越准确率：混淆矩阵、ROC 曲线、PR 曲线、错误分析（误报/漏报）。 |
| **🏭 工业级架构** | 流水线设计：`验证 → 预处理 → 训练 → 评估 → 部署`。 |
| **🔄 交叉验证** | 5 折交叉验证确保统计显著性，而非单次拆分的偶然性。 |
| **⚡ 生产就绪** | 压缩模型序列化，可直接用于 API 部署。 |

### 📊 实验结果

基于 PhiUSIIL Phishing URL Dataset 验证：

| 配置 | ROC-AUC | F1 分数 | 结论 |
| :--- | :---: | :---: | :--- |
| **朴素贝叶斯（基线）** | ~0.92 | ~0.91 | ✅ 快速、可解释 |
| **朴素贝叶斯（优化）** | **>0.95** | **>0.94** | ✅ 标准化 + 平滑处理 |
| **随机森林（100 棵树）** | **>0.98** | **>0.97** | ✅ 最佳综合性能 |
| **5 折交叉验证（RF）** | **0.97 ± 0.01** | **0.96 ± 0.01** | ✅ 统计稳健 |

### 🛠️ 快速开始

#### 安装

```bash
# 克隆代码库
git clone https://github.com/Mal-Suen/Malicious-Website-Detection.git
cd Malicious-Website-Detection

# 安装依赖
pip install -r requirements.txt
```

#### Python API

```python
from src.models.random_forest import RandomForestModel
from src.preprocess import get_raw_splits
from src.utils import evaluate_model

# 加载数据
X_train, X_test, y_train, y_test, raw_df = get_raw_splits()

# 初始化与训练
rf_model = RandomForestModel(n_estimators=100, max_depth=None)
rf_model.train(X_train, y_train)

# 评估
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_prob, model_name="Random Forest V1.0")

# 保存用于部署
rf_model.save_model("phish_rf_v1.pkl")
```

#### 命令行工具

```bash
# 运行朴素贝叶斯实验
python experiments/exp_nb.py

# 运行随机森林实验
python experiments/exp_rf.py

# 每个实验输出：
# - 综合指标（准确率、精确率、召回率、F1、AUC）
# - 可视化（混淆矩阵、ROC 曲线、PR 曲线）
# - 错误分析（误报/漏报）
# - 序列化的模型文件（saved_models/）
```

### 📂 目录结构

```text
Malicious-Website-Detection/
├── data/                          # 数据集
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
├── experiments/                   # 实验脚本
│   ├── exp_nb.py                  # 朴素贝叶斯工作流
│   └── exp_rf.py                  # 随机森林工作流
├── saved_models/                  # 序列化的模型
│   ├── phish_nb_v2.pkl            # 训练好的 NB 模型
│   └── phish_rf_v1.pkl            # 训练好的 RF 模型
├── src/                           # 核心 Python 包
│   ├── __init__.py                # 模块导出
│   ├── config.py                  # 配置与常量
│   ├── preprocess.py              # 数据加载与验证
│   ├── utils.py                   # 评估与可视化
│   ├── experiment_runner.py       # 通用实验流水线
│   └── models/                    # 模型实现
│       ├── __init__.py
│       ├── naive_bayes.py         # 高斯朴素贝叶斯分类器
│       └── random_forest.py       # 随机森林分类器
├── .gitignore                     # Git 忽略规则
├── README.md                      # 本文件
└── requirements.txt               # 依赖项
```

### 🔬 关键发现

1. **数据验证至关重要：** 包含缺失值的未验证数据集会使模型性能下降 5-10%。自动验证必须在训练之前执行。
2. **标准化不可或缺：** 即使是基于树的模型，特征缩放也能改善收敛性，并保持不同流水线变体之间的一致性。
3. **单次拆分不够充分：** 一次训练/测试拆分可能反映随机偶然性。需要使用 K 折交叉验证（K≥5）来确保统计显著性。
4. **错误分析驱动改进：** 分析误报/漏报揭示了特征工程的优化空间（如 URL 长度、域名熵值）。

---

## 🤝 Contribution & Contact / 贡献与联系

*   **Author:** Mal-Suen
*   **Blog:** [Mal-Suen's Blog](https://blog.mal-suen.cn)
*   **GitHub:** [https://github.com/Mal-Suen/Malicious-Website-Detection](https://github.com/Mal-Suen/Malicious-Website-Detection)

*Copyright © 2024-2026 Mal-Suen. Released under MIT License.*
