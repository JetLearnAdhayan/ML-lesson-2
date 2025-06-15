#multivariable regression
import numpy as np
import pandas as pd

data = pd.read_csv('titanic.csv')  
data['Sex']=data['Sex'].map({'male':0,'female':1})
print(data.head())
x = data[['Age','Sex','Pclass']]
y = data['Survived']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)

from sklearn.linear_model import LinearRegression
linearreg = LinearRegression()
#fit the model with traning data
linearreg.fit(x_train,y_train)

from sklearn.metrics import mean_squared_error

y_predict = linearreg.predict(x_test)

rmse_lr = (np.sqrt(mean_squared_error(y_test,y_predict)))

print("The rmse of MultiVariable Regression is:", rmse_lr)

#polynominal regression

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2)

x_train_poly = poly_features.fit_transform(x_train)

poly_model = LinearRegression()
poly_model.fit(x_train_poly,y_train)

x_test_poly = poly_features.fit_transform(x_test)
y_test_predict_poly = poly_model.predict(x_test_poly)

rmse_poly_model = (np.sqrt(mean_squared_error(y_test, y_test_predict_poly)))
print("the rmse in case of polynominial regression is", rmse_poly_model)