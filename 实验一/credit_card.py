from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

# 从ucimlrepo获取数据集
default_of_credit_card_clients = fetch_ucirepo(id=350)

# 数据（作为pandas数据帧）
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 应用SMOTE
sm = SMOTE(random_state=42)
# 对数据进行过采样，以解决类别不平衡问题
X_res, y_res = sm.fit_resample(X_train, y_train)

# 使用贝叶斯分类器
gnb = GaussianNB()

# 在原始数据上训练
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy_before = accuracy_score(y_test, y_pred)
f1_before = f1_score(y_test, y_pred)

# 在过采样数据上训练
gnb.fit(X_res, y_res)
y_res_pred = gnb.predict(X_test)
accuracy_after = accuracy_score(y_test, y_res_pred)
f1_after = f1_score(y_test, y_res_pred)

# 绘制图表
plt.bar(["Before SMOTE", "After SMOTE"], [f1_before, f1_after])
plt.ylabel("F1 Score")
plt.show()

# 绘制图表
plt.bar(["Before SMOTE", "After SMOTE"], [accuracy_before, accuracy_after])
plt.ylabel("Accuracy")
plt.show()

# 元数据
# print(default_of_credit_card_clients.metadata)

# 变量信息
# print(default_of_credit_card_clients.variables)
