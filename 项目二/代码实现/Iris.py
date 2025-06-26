# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签
feature_names = iris.feature_names
target_names = iris.target_names

# 将数据转换为DataFrame，方便可视化
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# 设置Seaborn风格
sns.set(style="whitegrid")

#散点图：花萼长度 vs 花萼宽度，按类别着色
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', style='species', data=df, s=100)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# 进行数据扩充操作
# 使用SMOTE生成合成样本
smote = SMOTE(sampling_strategy={0: 250, 1: 250, 2: 250}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# 重新变成DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=feature_names)
df_resampled['species'] = pd.Categorical.from_codes(y_resampled, target_names)

#散点图：花萼长度 vs 花萼宽度，按类别着色
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', style='species', data=df_resampled, s=100)
plt.title('Sepal Length vs Sepal Width')
plt.show()