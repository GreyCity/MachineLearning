import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1.导入数据
Molecular_Descriptor_training = pd.read_excel(r'Molecular_Descriptor.xlsx',sheet_name='training')  # 回归标签-训练
ER_activity_training = pd.read_excel(r'ERα_activity.xlsx',sheet_name='training')  # 回归数值-训练

X_train = Molecular_Descriptor_training.iloc[:, 1:]
y_train = ER_activity_training.iloc[:, 2]

print(X_train.shape)

# 2.方差选择法：去除方差为0的无效特征
vt = VarianceThreshold()
vt.fit(X_train)
mask_vt = vt.get_support()
X_train_vt = X_train.iloc[:, mask_vt]

print(X_train_vt.shape)

# 3.模型选择法：极限随机森林
etr = SelectFromModel(ExtraTreesRegressor())
etr.fit(X_train_vt, y_train)
mask_etr = etr.get_support()
X_train_etr = X_train_vt.iloc[:, mask_etr]

print(X_train_etr.shape)

# 4.LassoCV估计器：特征重要性
lcv = LassoCV(cv=10, max_iter=10000)
lcv.fit(X_train_etr, y_train)
importance = np.abs(lcv.coef_)
mask_lcv = (-importance).argsort()[:20]
X_train_lcv = X_train_etr.iloc[:, mask_lcv]
print(X_train_lcv.shape)

# 画特征重要性图
featureName = pd.DataFrame({"feature": X_train_etr.columns[mask_lcv]})
featureImportance = pd.DataFrame({"importance": importance[mask_lcv]})
featureData = pd.concat([featureName, featureImportance], axis=1)
print(featureData)
sns.barplot(x="importance", y="feature", data=featureData, orient="h")

# 5.保存数据

# result = pd.concat([Molecular_Descriptor_training.iloc[:, 0], X_train_lcv], axis=1)
# pd.DataFrame(result).to_excel('FeatureSelect.xlsx', index=0)
