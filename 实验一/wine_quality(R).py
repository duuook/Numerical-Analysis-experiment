import matplotlib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def wine_quality_analysis():
    """
    对葡萄酒质量数据集进行分析。

    此函数获取葡萄酒质量数据集，将其分为训练集和测试集，
    创建线性回归模型，将模型拟合到训练数据，对测试集进行预测，
    计算预测的均方误差，执行5折交叉验证，并绘制均方误差和交叉验证分数。

    返回:
        无
    """
    # 获取数据集
    wine_quality = fetch_ucirepo(id=186)

    # 数据（作为pandas数据帧）
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建线性回归模型
    model = LinearRegression()

    # 将模型拟合到训练数据
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算预测的均方误差
    mse = mean_squared_error(y_test, y_pred)

    # 执行5折交叉验证
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")

    # 将分数转为正数
    scores = -scores

    print(f"均方误差: {mse}")
    print(f"5折交叉验证分数: {scores}")
    print(f"5折交叉验证分数平均值: {scores.mean()}")

    # 绘图
    plt.figure(figsize=(10, 5))

    # 子图1：MSE
    plt.subplot(1, 2, 1)
    plt.bar(["MSE"], [mse])
    plt.title("均方误差")

    # 子图2：交叉验证分数
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(1, len(scores) + 1), scores)
    plt.title("5折交叉验证分数")
    plt.xlabel("折")
    plt.ylabel("分数")

    plt.tight_layout()
    plt.show()


def compare_regression_models(alpha=1.0):
    """
    对比线性回归和岭回归模型。

    此函数获取葡萄酒质量数据集，将其分为训练集和测试集，
    创建线性回归和岭回归模型，将模型拟合到训练数据，对测试集进行预测，
    计算预测的均方误差，并绘制均方误差比较图。

    返回:
        无
    """
    # 获取数据集
    wine_quality = fetch_ucirepo(id=186)

    # 数据（作为pandas数据帧）
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建线性回归模型
    model = LinearRegression()

    # 将模型拟合到训练数据
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算预测的均方误差
    mse = mean_squared_error(y_test, y_pred)

    # 创建岭回归模型
    ridge_model = Ridge(alpha=alpha)

    # 将岭回归模型拟合到训练数据
    ridge_model.fit(X_train, y_train)

    # 使用岭回归模型对测试集进行预测
    ridge_y_pred = ridge_model.predict(X_test)

    # 计算岭回归预测的均方误差
    ridge_mse = mean_squared_error(y_test, ridge_y_pred)

    print(f"线性回归的均方误差: {mse}")
    print(f"λ为 {alpha} 的岭回归的均方误差: {ridge_mse}")

    # 绘图
    plt.figure(figsize=(10, 5))

    # 子图1：均方误差比较
    plt.subplot(1, 2, 1)
    plt.bar(["线性回归", "岭回归"], [mse, ridge_mse])
    plt.title(f"λ为{alpha}的均方误差比较")

    plt.tight_layout()
    plt.show()


matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 调用函数,分别计算平方误差和平均值
wine_quality_analysis()
for i in [0.5, 1.0, 1.5]:
    compare_regression_models(alpha=i)
