from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib
import matplotlib.pyplot as plt

# fetch dataset
statlog_landsat_satellite = fetch_ucirepo(id=146)

# data (as pandas dataframes)
X = statlog_landsat_satellite.data.features
y = statlog_landsat_satellite.data.targets

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建一个高斯分类器
gnb = GaussianNB()

# 使用训练集训练模型
gnb.fit(X_train, y_train)

# 预测测试数据集的响应
y_pred = gnb.predict(X_test)

# SMOTE之前的模型F1分数
f1_before_smote = f1_score(y_test, y_pred, average="weighted")
print("SMOTE之前的F1分数: ", f1_before_smote)

# 应用SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 将重采样的数据分为训练集和测试集
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# 使用重采样的训练集训练模型
gnb.fit(X_train_res, y_train_res)

# 预测重采样的测试数据集的响应
y_pred_res = gnb.predict(X_test_res)

# SMOTE之后的模型F1分数
f1_after_smote = f1_score(y_test_res, y_pred_res, average="weighted")
print("SMOTE之后的F1分数: ", f1_after_smote)

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 绘制F1分数
plt.bar(["SMOTE之前", "SMOTE之后"], [f1_before_smote, f1_after_smote])
plt.xlabel("数据")
plt.ylabel("F1分数")
plt.title("F1分数比较")
plt.show()
