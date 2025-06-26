import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# 忽略警告以保持输出整洁
warnings.filterwarnings("ignore")

# 设置seaborn样式和调色板
sns.set(style='whitegrid')
palette = sns.color_palette("viridis", n_colors=2)

def preprocess_data(df):
    """预处理数据：编码目标变量，选择特征，计算相关性"""
    # 编码分类变量
    df = pd.get_dummies(df, columns=['Class'], drop_first=True)
    target = df['Class_Kecimen']
    target = target.astype(int)
    df['Class_Kecimen'] = df['Class_Kecimen'].astype(int)

    # 计算相关性矩阵
    corr_matrix = df.select_dtypes('number').corr()

    # 可视化相关性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='RdYlBu', annot=True)
    plt.title('相关性热图')
    plt.show()

    # 选择相关性绝对值>=0.6的特征
    selected_features = corr_matrix.index[abs(corr_matrix['Class_Kecimen']) >= 0.6].tolist()
    selected_features.remove('Class_Kecimen')
    if len(selected_features) > 2:
        selected_features = selected_features[:2]

    return df, target, selected_features, corr_matrix



def load_and_explore_data(file_path):
    """加载数据集并显示基本信息"""
    df = pd.read_excel(file_path)
    print("数据集概况：")
    df.info()
    print("\n数据集描述：")
    print(df.describe())
    print("\n类别计数：")
    print(df['Class'].value_counts())
    print("\n唯一类别：", df['Class'].unique())
    return df


def train_soft_margin_svm(X, y, C=1.0, gamma="scale", test_size=0.2, random_state=42):

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化特征（SVM 对特征尺度敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练软间隔 SVM 模型（使用 RBF 核）
    svm_model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=random_state)
    svm_model.fit(X_train_scaled, y_train)

    # 预测并评估模型
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n软间隔 SVM（RBF 核）结果：")
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    return svm_model, scaler, accuracy


def plot_decision_boundary(X, y, model, scaler, selected_features):
    """
    绘制二维特征空间的决策边界（适用于非线性 SVM）。

    参数:
        X (pd.DataFrame): 特征矩阵（需为 2 个特征）。
        y (pd.Series): 目标变量。
        model: 训练好的 SVM 模型。
        scaler: 已拟合的标准化器。
        selected_features (list): 选定特征的名称。
    """
    # 检查是否为二维特征
    if X.shape[1] != 2:
        print("决策边界可视化需要正好 2 个特征。")
        return

    # 标准化特征数据
    X_scaled = scaler.transform(X)

    # 创建网格点用于绘制决策边界
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 在网格点上进行预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和数据点
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="viridis", edgecolors="k", s=50
    )
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title("Soft Interval SVM (RBF Kernel) Decision Boundary")
    plt.show()

def main():
    """主函数，协调整体流程"""
    file_path = r"C:\Users\LENOVO\Desktop\work_ws\math_statistic\SVM_Pr\Raisin_Dataset.xlsx"

    # 加载并探索数据
    df = load_and_explore_data(file_path)

    # 预处理数据
    df, target, selected_features, corr_matrix = preprocess_data(df)

    # 准备特征数据
    X = df[selected_features]
    y = target
    # 训练并评估软间隔 SVM
    svm_model, scaler, accuracy = train_soft_margin_svm(X, y, C=1.0)

    # 绘制决策边界（如果选择了 2 个特征）
    plot_decision_boundary(X, y, svm_model, scaler, selected_features)
if __name__ == "__main__":
    main()