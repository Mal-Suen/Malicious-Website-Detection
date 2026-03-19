import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def evaluate_model(y_true, y_pred, y_prob, model_name="Model"):
    """
    输出分类报告并绘制混淆矩阵。
    """
    # --- 解决中文显示问题 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    print(f"\n--- {model_name} 评估报告 ---")
    # 生成精确率、召回率、F1分数
    print(classification_report(y_true, y_pred, target_names=['Phishing (0)', 'Legitimate (1)']))

    # 计算AUC得分
    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC 得分: {auc:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))

    # annot=True 显示数字，fmt='d' 格式化为整数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测-钓鱼(0)', '预测-合法(1)'],
                yticklabels=['实际-钓鱼(0)', '实际-合法(1)'])

    plt.title(f'混淆矩阵：{model_name}')
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')

    # 强制布局紧凑，防止文字溢出
    plt.tight_layout()
    plt.show()


def get_error_analysis(X_test, y_test, y_pred, original_df):
    """
    提取预测错误的原始样本进行追溯。
    """
    # 漏报：实际为0（钓鱼），预测为1（合法）
    fn_indices = X_test[(y_test == 0) & (y_pred == 1)].index
    # 误报：实际为1（合法），预测为0（钓鱼）
    fp_indices = X_test[(y_test == 1) & (y_pred == 0)].index

    return original_df.loc[fn_indices], original_df.loc[fp_indices]