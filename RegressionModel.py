import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

# 导入数据
Molecular_Descriptor_training = pd.read_excel(r'FeatureSelect.xlsx')
Molecular_Descriptor_detect = pd.read_excel(r'DetectSet.xlsx')
ER_activity_training = pd.read_excel(r'ERα_activity.xlsx',sheet_name='training')

X_train = Molecular_Descriptor_training.iloc[:, 1:]
X_detect = Molecular_Descriptor_detect.iloc[:, 1:]
y_train = ER_activity_training.iloc[:, 2]

# 划分数据
X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(X_train, y_train, test_size=0.25, random_state=33)

# 标准化
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train_svr = ss_X.fit_transform(X_train_svr)
X_test_svr = ss_X.transform(X_test_svr)
X_detect_svr = ss_X.transform(X_detect)

y_train_svr = ss_y.fit_transform(y_train_svr.values.reshape(-1, 1))
y_test_svr = ss_y.transform(y_test_svr.values.reshape(-1, 1))


# SVR
# 使用三种不同核函数配置的支持向量机回归模型进行训练，并且分别对测试数据进行预测

#1.使用线性核函数配置的支持向量机进行回归训练并预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train_svr,y_train_svr)
linear_svr_y_predict = linear_svr.predict(X_test_svr)

#2.使用多项式核函数配置的支持向量机进行回归训练并预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train_svr,y_train_svr)
poly_svr_y_predict = poly_svr.predict(X_test_svr)

#3.使用径向基核函数配置的支持向量机进行回归训练并预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train_svr,y_train_svr)
rbf_svr_y_predict = rbf_svr.predict(X_test_svr)

#第五步：对三种核函数配置下的支持向量机回归模型在相同测试集下进行性能评估
#使用R-squared、MSE、MAE指标评估

#1.线性核函数配置的SVR
print('R-squared value of linear SVR is',linear_svr.score(X_test_svr,y_test_svr))
print('the MSE of linear SVR is',mean_squared_error(ss_y.inverse_transform(y_test_svr),ss_y.inverse_transform(linear_svr_y_predict)))
print('the MAE of linear SVR is',mean_absolute_error(ss_y.inverse_transform(y_test_svr),ss_y.inverse_transform(linear_svr_y_predict)))

#2.多项式核函数配置的SVR
print('R-squared value of Poly SVR is',poly_svr.score(X_test_svr,y_test_svr))
print('the MSE of Poly SVR is',mean_squared_error(ss_y.inverse_transform(y_test_svr),ss_y.inverse_transform(poly_svr_y_predict)))
print('the MAE of Poly SVR is',mean_absolute_error(ss_y.inverse_transform(y_test_svr),ss_y.inverse_transform(poly_svr_y_predict)))

#3.径向基核函数配置的SVR
print('R-squared value of RBF SVR is',rbf_svr.score(X_test_svr,y_test_svr))
print('the MSE of RBF SVR is',mean_squared_error(ss_y.inverse_transform(y_test_svr),ss_y.inverse_transform(rbf_svr_y_predict)))
print('the MAE of RBF SVR is',mean_absolute_error(ss_y.inverse_transform(y_test_svr),ss_y.inverse_transform(rbf_svr_y_predict)))

# 预测
y_detect_svr = rbf_svr.predict(X_detect_svr)
y_detect = ss_y.inverse_transform(y_detect_svr)
print(y_detect.shape)
y_detect = pd.DataFrame(y_detect)
result = pd.concat([Molecular_Descriptor_detect.iloc[:, 0], y_detect], axis=1)
pd.DataFrame(result).to_excel('RegressionForecasting.xlsx', index=0)