# 🛡️ Malicious Website Detection / 恶意网站检测

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A machine learning project for detecting phishing and malicious websites using multiple algorithms.**

**一个使用多种算法检测钓鱼和恶意网站的机器学习项目。**

---

## 📑 Table of Contents / 目录

- [English Version](#english-version)
  - [Overview](#overview)
  - [Features](#features)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Project Structure](#project-structure)
  - [Model Architecture](#model-architecture)
  - [Configuration](#configuration)
  - [Results](#results)
  - [Improvements](#improvements)
  - [Future Work](#future-work)
  - [License](#license)

- [中文版本](#中文版本)
  - [项目简介](#项目简介)
  - [核心功能](#核心功能)
  - [数据集](#数据集)
  - [安装指南](#安装指南)
  - [快速开始](#快速开始)
  - [项目结构](#项目结构)
  - [模型架构](#模型架构)
  - [配置说明](#配置说明)
  - [实验结果](#实验结果)
  - [改进内容](#改进内容)
  - [未来工作](#未来工作)
  - [许可证](#许可证)

---

## English Version

### Overview

Malicious Website Detection is a machine learning project that aims to identify phishing and malicious URLs automatically. This project implements multiple classification algorithms (Naive Bayes and Random Forest) to compare their performance in detecting malicious websites. The codebase follows best practices with proper logging, type hints, cross-validation, and comprehensive error analysis.

### Features

- **Multi-Algorithm Support**: Gaussian Naive Bayes and Random Forest classifiers
- **Automated Data Validation**: Validates dataset integrity and handles missing values automatically
- **K-Fold Cross Validation**: Provides robust model evaluation with cross-validation
- **Comprehensive Evaluation**: Includes ROC curves, PR curves, confusion matrix, and detailed classification reports
- **Error Analysis**: Identifies false positives and false negatives for model improvement
- **Pipeline Architecture**: Encapsulates preprocessing and modeling using sklearn Pipeline
- **Logging System**: Professional logging instead of print statements
- **Cross-Platform**: Automatically adapts to Windows/macOS/Linux environments
- **Type Hints**: Complete type annotations for better IDE support

### Dataset

This project uses the **PhiUSIIL Phishing URL Dataset**, which contains:
- URL features and metadata
- Binary labels (0 = Phishing, 1 = Legitimate)
- Multiple numerical features for classification

**Data file**: `data/PhiUSIIL_Phishing_URL_Dataset.csv`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mal-Suen/Malicious-Website-Detection.git
   cd Malicious-Website-Detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

**Run Naive Bayes experiment**:
```bash
python experiments/exp_nb.py
```

**Run Random Forest experiment**:
```bash
python experiments/exp_rf.py
```

Each experiment will:
1. Load and validate the dataset
2. Split data into training and testing sets (80/20)
3. Train the model with preprocessing pipeline
4. Evaluate on test set with comprehensive metrics
5. Perform 5-fold cross-validation
6. Conduct error analysis (false positives/negatives)
7. Save the trained model to `saved_models/`

### Project Structure

```
Malicious-Website-Detection/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv   # Dataset
├── experiments/
│   ├── exp_nb.py                           # Naive Bayes experiment
│   └── exp_rf.py                           # Random Forest experiment
├── saved_models/
│   ├── phish_nb_v2.pkl                     # Trained NB model
│   └── phish_rf_v1.pkl                     # Trained RF model
├── src/
│   ├── models/
│   │   ├── naive_bayes.py                  # NB model implementation
│   │   └── random_forest.py                # RF model implementation
│   ├── config.py                           # Configuration and constants
│   ├── preprocess.py                       # Data loading and validation
│   ├── utils.py                            # Evaluation and visualization
│   ├── experiment_runner.py                # Generic experiment runner
│   └── __init__.py                         # Module exports
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

### Model Architecture

Both models follow a unified architecture:

1. **Data Preprocessing**: 
   - Feature extraction (removing metadata columns)
   - Missing value handling (median for numerical, empty string for text)
   - Data validation and integrity checks

2. **Model Pipeline**:
   - `StandardScaler`: Feature standardization
   - `Classifier`: GaussianNB or RandomForestClassifier

3. **Training**:
   - Stratified train/test split (80/20)
   - Fixed random state for reproducibility (RANDOM_STATE=42)

4. **Evaluation**:
   - Classification report (Precision, Recall, F1-Score)
   - ROC-AUC Score
   - Average Precision Score
   - Confusion Matrix visualization
   - ROC and PR curves

### Configuration

Key parameters in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RANDOM_STATE` | 42 | Fixed random seed for reproducibility |
| `TEST_SIZE` | 0.2 | Test set ratio (20%) |
| `CV_FOLDS` | 5 | Cross-validation folds |
| `LABEL_COL` | 'label' | Target column name |
| `METADATA_COLS` | ['FILENAME', 'URL', ...] | Columns excluded from features |

### Results

After running experiments, you will see:
- Detailed classification metrics in console
- Three-panel visualization (Confusion Matrix, ROC Curve, PR Curve)
- Error analysis with false positive/negative samples
- Saved model files ready for deployment

### Improvements

This codebase includes the following improvements over the initial version:
- ✅ Data validation and missing value handling
- ✅ Cross-platform font compatibility for visualizations
- ✅ Complete type hints and docstrings
- ✅ Logging system replacing print statements
- ✅ K-Fold cross-validation support
- ✅ Generic experiment runner to eliminate code duplication
- ✅ Enhanced evaluation metrics (ROC + PR curves)
- ✅ Unified configuration management
- ✅ Proper module exports in `__init__.py`
- ✅ `.gitignore` for clean repository

### Future Work

- [ ] Add more algorithms (XGBoost, LightGBM, Neural Networks)
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Add feature importance analysis
- [ ] Create a web API for real-time prediction
- [ ] Support online learning and model updates
- [ ] Add unit tests and integration tests

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 中文版本

### 项目简介

恶意网站检测是一个旨在自动识别钓鱼和恶意 URL 的机器学习项目。本项目实现了多种分类算法（朴素贝叶斯和随机森林）来比较它们在检测恶意网站方面的性能。代码库遵循最佳实践，包含完善的日志系统、类型注解、交叉验证和全面的错误分析。

### 核心功能

- **多算法支持**：高斯朴素贝叶斯和随机森林分类器
- **自动数据验证**：验证数据集完整性并自动处理缺失值
- **K 折交叉验证**：提供稳健的模型评估
- **全面评估指标**：包含 ROC 曲线、PR 曲线、混淆矩阵和详细分类报告
- **错误分析**：识别误报和漏报以便改进模型
- **Pipeline 架构**：使用 sklearn Pipeline 封装预处理和建模
- **日志系统**：专业的 logging 模块替代 print 语句
- **跨平台兼容**：自动适配 Windows/macOS/Linux 环境
- **类型注解**：完整的类型提示，更好的 IDE 支持

### 数据集

本项目使用 **PhiUSIIL Phishing URL Dataset**，包含：
- URL 特征和元数据
- 二分类标签（0 = 钓鱼网站，1 = 合法网站）
- 多个用于分类的数值型特征

**数据文件**：`data/PhiUSIIL_Phishing_URL_Dataset.csv`

### 安装指南

1. **克隆代码库**：
   ```bash
   git clone https://github.com/Mal-Suen/Malicious-Website-Detection.git
   cd Malicious-Website-Detection
   ```

2. **创建虚拟环境**（推荐）：
   ```bash
   python -m venv .venv
   # Windows 系统
   .venv\Scripts\activate
   # macOS/Linux 系统
   source .venv/bin/activate
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

### 快速开始

**运行朴素贝叶斯实验**：
```bash
python experiments/exp_nb.py
```

**运行随机森林实验**：
```bash
python experiments/exp_rf.py
```

每个实验将会：
1. 加载并验证数据集
2. 将数据拆分为训练集和测试集（80/20）
3. 使用预处理流水线训练模型
4. 使用全面指标评估测试集
5. 执行 5 折交叉验证
6. 进行错误分析（误报/漏报）
7. 将训练好的模型保存到 `saved_models/`

### 项目结构

```
Malicious-Website-Detection/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv   # 数据集
├── experiments/
│   ├── exp_nb.py                           # 朴素贝叶斯实验
│   └── exp_rf.py                           # 随机森林实验
├── saved_models/
│   ├── phish_nb_v2.pkl                     # 训练好的 NB 模型
│   └── phish_rf_v1.pkl                     # 训练好的 RF 模型
├── src/
│   ├── models/
│   │   ├── naive_bayes.py                  # NB 模型实现
│   │   └── random_forest.py                # RF 模型实现
│   ├── config.py                           # 配置和常量
│   ├── preprocess.py                       # 数据加载和验证
│   ├── utils.py                            # 评估和可视化
│   ├── experiment_runner.py                # 通用实验运行器
│   └── __init__.py                         # 模块导出
├── requirements.txt                        # Python 依赖
└── README.md                               # 本文件
```

### 模型架构

两个模型遵循统一的架构：

1. **数据预处理**：
   - 特征提取（移除元数据列）
   - 缺失值处理（数值列用中位数，文本列用空字符串）
   - 数据验证和完整性检查

2. **模型流水线**：
   - `StandardScaler`：特征标准化
   - `Classifier`：GaussianNB 或 RandomForestClassifier

3. **训练过程**：
   - 分层训练/测试拆分（80/20）
   - 固定随机种子保证可复现性（RANDOM_STATE=42）

4. **评估指标**：
   - 分类报告（精确率、召回率、F1 分数）
   - ROC-AUC 得分
   - 平均精确度得分
   - 混淆矩阵可视化
   - ROC 和 PR 曲线

### 配置说明

`src/config.py` 中的关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `RANDOM_STATE` | 42 | 固定随机种子保证可复现性 |
| `TEST_SIZE` | 0.2 | 测试集比例（20%） |
| `CV_FOLDS` | 5 | 交叉验证折数 |
| `LABEL_COL` | 'label' | 目标列名称 |
| `METADATA_COLS` | ['FILENAME', 'URL', ...] | 从特征中排除的列 |

### 实验结果

运行实验后，您将看到：
- 控制台中详细的分类指标
- 三面板可视化（混淆矩阵、ROC 曲线、PR 曲线）
- 包含误报/漏报样本的错误分析
- 保存的模型文件可用于部署

### 改进内容

相比初始版本，本代码库包含以下改进：
- ✅ 数据验证和缺失值处理
- ✅ 可视化跨平台字体兼容性
- ✅ 完整的类型注解和文档字符串
- ✅ 日志系统替代 print 语句
- ✅ K 折交叉验证支持
- ✅ 通用实验运行器消除代码重复
- ✅ 增强的评估指标（ROC + PR 曲线）
- ✅ 统一的配置管理
- ✅ `__init__.py` 中的正确模块导出
- ✅ `.gitignore` 保持代码库整洁

### 未来工作

- [ ] 添加更多算法（XGBoost、LightGBM、神经网络）
- [ ] 使用 GridSearchCV 实现超参数调优
- [ ] 添加特征重要性分析
- [ ] 创建用于实时预测的 Web API
- [ ] 支持在线学习和模型更新
- [ ] 添加单元测试和集成测试

### 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE) 文件。

---

## 📧 Contact / 联系方式

For questions or suggestions, please open an issue or contact the repository owner.

如有问题或建议，请提 Issue 或联系代码库所有者。
