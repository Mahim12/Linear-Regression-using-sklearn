import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
  
# Changing the file read location to the location of the dataset
df = pd.read_csv('diamonds.csv')
print(df.head())

x = df["carat"] #features
y = df["price"] #label: The target 

plt.scatter(x, y)
plt.savefig('overview.png')
plt.show()
  

X = np.array(df["carat"]).reshape(-1, 1)
print(X)
y = np.array(df["price"]).reshape(-1, 1)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
  
# Splitting the data into training and testing data
regr = LinearRegression()
  
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))


y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.savefig('prediction with best fit.png')    
plt.show()
