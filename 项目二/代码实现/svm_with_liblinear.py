import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# 忽略警告信息以保持输出整洁
warnings.filterwarnings("ignore")

# 设置 Seaborn 可视化风格和颜色调色板
sns.set(style="whitegrid")
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


class CoordinateDescentSVM:
    """
    使用坐标下降法实现的软间隔线性 SVM，参考 LIBLINEAR。

    参数:
        C (float): 正则化参数，控制软间隔大小。
        tol (float): 收敛容差。
        max_iter (int): 最大迭代次数。
    """

    def __init__(self, C=1.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None  # 拉格朗日乘子
        self.w = None  # 权重向量
        self.b = None  # 偏置项
        self.support_vectors_ = None

    def fit(self, X, y):
        """
        训练 SVM 模型，使用坐标下降法优化对偶问题。

        参数:
            X (np.ndarray): 特征矩阵，形状 (n_samples, n_features)。
            y (np.ndarray): 目标变量，值为 {-1, 1}。  你
        """
        n_samples, n_features = X.shape
        y = np.where(y == 0, -1, 1)  # 转换为 {-1, 1} 标签
        self.alpha = np.zeros(n_samples)  # 初始化拉格朗日乘子
        Xy = X * y[:, np.newaxis]  # 预计算 X * y，优化计算
        Q_diag = np.sum(X ** 2, axis=1)  # Q 矩阵对角线元素：x_i^T x_i

        # 坐标下降法主循环
        for iter_ in range(self.max_iter):
            max_diff = 0  # 记录 alpha 变化的最大值，用于收敛判断
            for i in range(n_samples):
                # 计算梯度 G_i = 1 - y_i * (w^T x_i)
                w_dot_x = np.dot(X[i], np.sum(Xy * self.alpha[:, np.newaxis], axis=0))
                G_i = 1 - y[i] * w_dot_x

                # 保存旧的 alpha 值
                alpha_old = self.alpha[i]

                # 更新 alpha_i
                self.alpha[i] = min(max(self.alpha[i] - G_i / (Q_diag[i] + 1e-8), 0), self.C)

                # 计算 alpha 变化量
                diff = abs(self.alpha[i] - alpha_old)
                max_diff = max(max_diff, diff)

            # 检查收敛
            if max_diff < self.tol:
                print(f"坐标下降法在第 {iter_ + 1} 次迭代收敛")
                break
        else:
            print(f"达到最大迭代次数 {self.max_iter}，未完全收敛")

        # 计算权重向量 w = sum(alpha_i * y_i * x_i)
        self.w = np.sum(self.alpha[:, np.newaxis] * Xy, axis=0)

        # 选择支持向量 (0 < alpha_i < C)
        sv_idx = (self.alpha > 0) & (self.alpha < self.C)
        self.support_vectors_ = X[sv_idx]

        # 计算偏置项 b，使用支持向量的平均值
        if np.sum(sv_idx) > 0:
            self.b = np.mean(y[sv_idx] - np.dot(self.support_vectors_, self.w))
        else:
            self.b = 0

        return self

    def predict(self, X):
        """
        预测样本的类别。

        参数:
            X (np.ndarray): 特征矩阵。

        返回:
            np.ndarray: 预测的类别标签 {0, 1}。
        """
        # 计算 f(x) = w^T x + b
        decision = np.dot(X, self.w) + self.b
        return np.where(decision >= 0, 1, 0)


def train_coordinate_descent_svm(X, y, C=1.0, test_size=0.2, random_state=42):
    """
    训练坐标下降法实现的软间隔 SVM，并评估性能。

    参数:
        X (pd.DataFrame): 特征矩阵。
        y (pd.Series): 目标变量。
        C (float): 正则化参数。
        test_size (float): 测试集比例。
        random_state (int): 随机种子。

    返回:
        tuple: (自定义 SVM 模型, 标准化器, 测试集准确率)
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练自定义坐标下降法 SVM
    svm_model = CoordinateDescentSVM(C=C)
    svm_model.fit(X_train_scaled, y_train)

    # 预测并评估
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n自定义坐标下降法 SVM 结果：")
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    return svm_model, scaler, accuracy


def plot_decision_boundary(X, y, model, scaler, selected_features):
    """
    绘制二维特征空间的决策边界，并显示明确的边界线。

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

    # 创建网格点用于绘制分类区域
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 预测网格点类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制分类区域和数据点
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")  # 分类区域
    plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="viridis", edgecolors="k", s=50
    )  # 数据点

    # 绘制决策边界线 w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
    w = model.w
    b = model.b
    x1 = np.array([x_min, x_max])
    x2 = -(w[0] * x1 + b) / (w[1] + 1e-8)  # 避免除零
    plt.plot(x1, x2, "k-", linewidth=2, label="Decision Boundary")  # 黑色实线表示边界

    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title("Soft Interval SVM (Coordinate Descent Method) Decision Boundary")
    plt.legend()
    plt.show()


def main():
    """
    主函数，协调数据处理和非线性 SVM 分类流程。
    """
    # 指定数据集路径
    file_path = r"C:\Users\LENOVO\Desktop\work_ws\math_statistic\SVM_Pr\Raisin_Dataset.xlsx"

    # 加载并探索数据
    df = load_and_explore_data(file_path)

    # 预处理数据
    df, target, selected_features, corr_matrix = preprocess_data(df)

    # 准备特征矩阵和目标变量
    X = df[selected_features]
    y = target

    # 训练并评估坐标下降法 SVM
    svm_model, scaler, accuracy = train_coordinate_descent_svm(X, y, C=0.01)

    # 绘制决策边界（如果选择了 2 个特征）
    plot_decision_boundary(X, y, svm_model, scaler, selected_features)

if __name__ == "__main__":
    main()