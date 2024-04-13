import matplotlib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def perform_ridge_regression(X, y):
    """
    执行岭回归并绘制结果图形的函数。

    参数:
    X: 特征数据
    y: 目标数据

    返回值:
    无
    """
    # 创建一个岭回归模型
    model = Ridge()

    # 拟合模型
    model.fit(X, y)

    # 预测目标
    y_pred = model.predict(X)

    # 计算均方误差
    mse = mean_squared_error(y, y_pred)

    # 执行5折交叉验证
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")

    # 将分数转换为正数
    scores = -scores

    print(f"岭回归均方误差: {mse}")
    print(f"岭回归5折交叉验证均方误差: {scores.mean()}")

    # 创建一个图形和轴
    fig, ax = plt.subplots()

    # 设置标签
    ax.set_xlabel("折叠")
    ax.set_ylabel("均方误差")

    # 为每个折叠绘制MSE
    ax.plot(np.arange(1, 6), scores, marker="o", linestyle="-", color="b")

    # 绘制平均MSE
    ax.axhline(scores.mean(), color="r", linestyle="--", label="岭回归平均")

    # 显示图例
    ax.legend()

    # 显示图形
    plt.show()


def perform_linear_regression(X, y):
    """
    执行线性回归并绘制结果图形的函数。

    参数:
    X: 特征数据
    y: 目标数据

    返回值:
    无
    """
    # Convert the categorical feature 'Sex' into numerical values
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])

    # 创建一个线性回归模型
    model = LinearRegression()

    # 拟合模型
    model.fit(X, y)

    # 预测目标
    y_pred = model.predict(X)

    # 计算均方误差
    mse = mean_squared_error(y, y_pred)

    # 执行5折交叉验证
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")

    # 将分数转换为正数
    scores = -scores

    print(f"均方误差: {mse}")
    print(f"5折交叉验证均方误差: {scores.mean()}")

    # 创建一个图形和轴
    fig, ax = plt.subplots()

    # 设置标签
    ax.set_xlabel("折叠")
    ax.set_ylabel("均方误差")

    # 为每个折叠绘制MSE
    ax.plot(np.arange(1, 6), scores, marker="o", linestyle="-", color="b")

    # 绘制平均MSE
    ax.axhline(scores.mean(), color="r", linestyle="--", label="平均")

    # 显示图例
    ax.legend()

    # 显示图形
    plt.show()


# fetch dataset
abalone = fetch_ucirepo(id=1)

# data (as pandas dataframes)
X = abalone.data.features
y = abalone.data.targets

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# Perform linear regression
perform_linear_regression(X, y)

# Perform ridge regression
perform_ridge_regression(X, y)
