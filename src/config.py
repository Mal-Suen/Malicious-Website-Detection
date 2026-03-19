import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models')

# 实验参数配置
RANDOM_STATE = 42  # 随机种子，固定此数值可保证每次运行的数据拆分结果一致
TEST_SIZE = 0.2    # 测试集占比 20%

# 数据列配置
# 标签列：1代表合法，0代表钓鱼
LABEL_COL = 'label'
# 元数据列：不参与数学计算的文本信息
METADATA_COLS = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']