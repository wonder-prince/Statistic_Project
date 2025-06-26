import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# SVM硬间隔
from sklearn.svm import LinearSVC
import warnings

# 忽略警告以保持输出整洁
warnings.filterwarnings("ignore")

# 设置seaborn样式和调色板
sns.set(style='whitegrid')
palette = sns.color_palette("viridis", n_colors=2)




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


def visualize_pairplot(df):
    """创建并自定义成对关系图"""
    pairplot = sns.pairplot(data=df, corner=True, hue='Class', palette='viridis')
    pairplot.fig.suptitle('自定义成对关系图', y=1.02)
    pairplot._legend.set_title('类别')

    for ax in pairplot.axes.flat:
        if ax is not None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles=handles[:2],
                    labels=labels[:2],
                    loc="upper right",
                    title="类别",
                    fontsize="small",
                    title_fontsize="medium"
                )
    plt.show()


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
    plt.title('correlation heat map')
    plt.show()

    # 选择相关性绝对值>=0.6的特征
    selected_features = corr_matrix.index[abs(corr_matrix['Class_Kecimen']) >= 0.6].tolist()
    selected_features.remove('Class_Kecimen')
    if len(selected_features) > 2:
        selected_features = selected_features[:2]

    return df, target, selected_features, corr_matrix


def train_logistic_model(X, y):
    """训练逻辑回归模型并返回预测结果"""
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # 预测并评估
    y_pred = model.predict(X_test_scaled)
    score = model.score(X_test_scaled, y_test)

    print("\n逻辑回归模型系数：", model.coef_)
    print("测试集预测结果：", y_pred)
    print("测试集准确率：", score)

    return model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred


def train_svm_model(X, y):
    """训练硬间隔SVM模型并返回预测结果"""
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练硬间隔SVM模型（C设为很大值以模拟硬间隔）
    try:
        model = LinearSVC(C=1e10, max_iter=10000)
        model.fit(X_train_scaled, y_train)
    except ValueError as e:
        print("错误：数据可能不是线性可分的，无法使用硬间隔SVM。", e)
        return None, None, None, None, None, None, None

    # 预测
    y_pred = model.predict(X_test_scaled)
    score = model.score(X_test_scaled, y_test)
    print("\n硬间隔SVM回归模型系数：", model.coef_)
    print("测试集预测结果：", y_pred)
    print("测试集准确率：", score)

    return model, X_test, X_test_scaled, X_train,X_train_scaled, y_train, y_test, y_pred, scaler


def visualize_decision_boundary(model, X_test, y_test, y_pred, selected_features):
    """可视化决策边界和测试数据点"""
    # 如果需要，将标准化数据转换为numpy数组
    X_test_scaled_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

    # 创建用于可视化的DataFrame
    df_test = pd.DataFrame(X_test, columns=selected_features)
    df_test["True_Class"] = pd.Series(y_test).map({0: "Besni", 1: "Kecimen"})
    df_test["Pred_Class"] = pd.Series(y_pred).map({0: "Besni", 1: "Kecimen"})
    df_test["Misclassified"] = df_test["True_Class"] != df_test["Pred_Class"]

    # 创建网格用于决策边界
    x_min, x_max = X_test_scaled_np[:, 0].min() - 1, X_test_scaled_np[:, 0].max() + 1
    y_min, y_max = X_test_scaled_np[:, 1].min() - 1, X_test_scaled_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(Z == 0, 0, 1)
    Z = Z.reshape(xx.shape)

    # 决策边界
    # plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis", levels=[-0.5, 0.5, 1.5])
    # plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis", levels=[-0.5, 0.5, 1.5])
    sns.scatterplot(
        data=df_test,
        x=selected_features[0],
        y=selected_features[1],
        hue="True_Class",
        style="Misclassified",
        style_order=[False, True],
        markers={False: "o", True: "X"},
        palette=palette,
        s=100,
    )
    plt.title(f"test_data:logistics_boundary\n{selected_features[0]} vs {selected_features[1]}")
    plt.legend(title="class")
    plt.show()


def visualize_decision_boundary_for_svm(model, X_test, X_test_scaled, y_test, y_pred, selected_features, scaler):
    """Visualize the decision boundary (straight line) and support vectors for hard-margin SVM on the test set"""
    if model is None:
        print("Cannot visualize: Model training failed.")
        return

    # Create DataFrame for visualization (using unscaled features)
    df_test = pd.DataFrame(X_test, columns=selected_features)
    df_test["True Class"] = pd.Series(y_test).map({0: "Besni", 1: "Kecimen"})
    df_test["Predicted Class"] = pd.Series(y_pred).map({0: "Besni", 1: "Kecimen"})
    df_test["Misclassified"] = df_test["True Class"] != df_test["Predicted Class"]

    # Create grid for decision boundary (based on unscaled features)
    x_min, x_max = X_test[selected_features[0]].min() - 1, X_test[selected_features[0]].max() + 1
    y_min, y_max = X_test[selected_features[1]].min() - 1, X_test[selected_features[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                         np.arange(y_min, y_max, (y_max - y_min) / 100))

    # Scale grid points for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.figure(figsize=(10, 8))
    # Plot decision boundary (straight line)
    plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles=['-'])
    # Plot margin boundaries (w*x + b = ±1)
    Z_margin = model.decision_function(grid_points_scaled)
    Z_margin = Z_margin.reshape(xx.shape)
    plt.contour(xx, yy, Z_margin, levels=[-1, 1], colors='k', linestyles=['--'])

    # Plot test set scatter plot
    sns.scatterplot(
        data=df_test,
        x=selected_features[0],
        y=selected_features[1],
        hue='True Class',
        style='Misclassified',
        style_order=[False, True],
        markers={False: 'o', True: 'X'},
        palette=palette,
        s=100
    )

    # Plot test set support vectors
    distances = model.decision_function(X_test_scaled)  # Modified here to decide whether to use test or training set
    support_vectors = X_test[np.abs(distances) <= 1 + 1e-6]
    plt.scatter(support_vectors[selected_features[0]],
                support_vectors[selected_features[1]],
                s=100, facecolors='none', edgecolors='red', label='Support Vectors')

    plt.title(f"Test Set Data: Hard-Margin SVM Decision Boundary\n{selected_features[0]} vs {selected_features[1]}")
    plt.legend(title="Class")
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.show()



def main():
    """主函数，协调整体流程"""
    file_path = r"C:\Users\LENOVO\Desktop\work_ws\math_statistic\SVM_Pr\Raisin_Dataset.xlsx"

    # 加载并探索数据
    df = load_and_explore_data(file_path)

    # 可视化成对关系图
    visualize_pairplot(df)

    # 预处理数据
    df, target, selected_features, corr_matrix = preprocess_data(df)

    # 准备特征数据
    X = df[selected_features]
    y = target

    # 训练Logistic Regression模型并预测
    model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred = train_logistic_model(X, y)
    # 可视化决策边界
    visualize_decision_boundary(model, X_test_scaled, y_test, y_pred, selected_features)

    # 训练硬间隔SVM模型并预测
    model, X_test, X_test_scaled, X_train,X_train_scaled, y_train, y_test, y_pred, scaler = train_svm_model(X, y)
    visualize_decision_boundary_for_svm(model, X_test, X_test_scaled, y_test, y_pred, selected_features, scaler)
    visualize_decision_boundary_for_svm(model, X_train, X_train_scaled, y_train, y_pred, selected_features, scaler)


if __name__ == "__main__":
    main()
