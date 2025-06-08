#multivariable regression
import numpy as np
import pandas as pd

data = pd.read_csv("titanic.csv")
print(data.head())

data = data["Sex"]=data["Sex"].map({"male": 0,"female":1})
print(data.head())
#print(data["Age"])

x = data[['Age','Sex','Pclass']]
y = data["Survived"]

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