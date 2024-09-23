
import numpy as np
import pandas as pd


 
data=pd.read_csv("winequality-red.csv")

  
# # Standardizing the data & gaining insights from the data

 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

 
data.info()

 
unique_counts = data.nunique()
unique_counts

 
data.head()

 
data.describe()

 
X=data.drop("quality",axis=1)

 
Y=data["quality"]

 
X.head()

 
Y.head()

   
# # Data Visualization

 
import seaborn as sns
import matplotlib.pyplot as plt


 
plt.figure(figsize=(15,8))
sns.heatmap(data.corr(),annot=True)

   
# # splitting the data into train and test data

 
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,stratify=Y, random_state=42, test_size=0.2)

   
# # Linear Regression model

 
from sklearn.linear_model import LinearRegression

 
linear=LinearRegression()
linear.fit(X_train,Y_train)

 
user_input=(8.1,0.56,0.28,1.7,0.368,16.0,56.0,0.9968,3.11,1.28,9.3

)
#changing the user_input to numpy array
userInputArray=np.asarray(user_input)
#reshaping the numpy array
userInputReshaped=userInputArray.reshape(1,-1)
prediction=linear.predict(userInputReshaped)
print(prediction)

   
# # Ridge Regression

 
from sklearn.linear_model import Ridge
ridge=Ridge(alpha=1,solver="cholesky")
ridge.fit(X_train,Y_train)


 
user_input=(8.1,0.56,0.28,1.7,0.368,16.0,56.0,0.9968,3.11,1.28,9.3

)
#changing the user_input to numpy array
userInputArray=np.asarray(user_input)
#reshaping the numpy array
userInputReshaped=userInputArray.reshape(1,-1)
prediction=ridge.predict(userInputReshaped)
print(prediction)

   
# # Stochastic Gradient Descent model

 
from sklearn.linear_model import SGDRegressor

 
#penalty="l2": This applies L2 regularization, which helps prevent overfitting by penalizing large coefficients.

sgd=SGDRegressor(penalty="l2")
sgd.fit(X_train,Y_train)


 
user_input=(8.1,0.56,0.28,1.7,0.368,16.0,56.0,0.9968,3.11,1.28,9.3

)
#changing the user_input to numpy array
userInputArray=np.asarray(user_input)
#reshaping the numpy array
userInputReshaped=userInputArray.reshape(1,-1)
prediction=sgd.predict(userInputReshaped)
print(prediction)

   
# # Lasso Regression

 
from sklearn.linear_model import Lasso

 
lassoReg=Lasso(alpha=0.1)
lassoReg.fit(X_train,Y_train)

 
user_input=(8.1,0.56,0.28,1.7,0.368,16.0,56.0,0.9968,3.11,1.28,9.3

)
#changing the user_input to numpy array
userInputArray=np.asarray(user_input)
#reshaping the numpy array
userInputReshaped=userInputArray.reshape(1,-1)
prediction=lassoReg.predict(userInputReshaped)
print(prediction)

   
# # ElasticNet Regression

 
from sklearn.linear_model import ElasticNet

 
elasticNet=ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticNet.fit(X_train,Y_train)

 
user_input=(8.1,0.56,0.28,1.7,0.368,16.0,56.0,0.9968,3.11,1.28,9.3

)
#changing the user_input to numpy array
userInputArray=np.asarray(user_input)
#reshaping the numpy array
userInputReshaped=userInputArray.reshape(1,-1)
prediction=elasticNet.predict(userInputReshaped)
print(prediction)

   
# # we have used Linear regression and various other shinkage regression methods and have got almost the same answer

   
# #Simple definition
# 1. Lasso Regression
# Definition: Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that uses L1 regularization.
# Key Feature: It can shrink some coefficients to zero, effectively performing variable selection.
# Use Case: Useful when you want a simpler model with fewer predictors.
# 2. Ridge Regression
# Definition: Ridge regression is a linear regression technique that employs L2 regularization.
# Key Feature: It penalizes large coefficients but does not reduce them to zero, keeping all predictors in the model.
# Use Case: Useful when you have multicollinearity or want to prevent overfitting.
# 3. Elastic Net
# Definition: Elastic Net combines both L1 and L2 regularization.
# Key Feature: It balances the benefits of Lasso (feature selection) and Ridge (handling multicollinearity).
# Use Case: Effective when dealing with datasets where the number of predictors is much larger than the number of observations.
# 4. Stochastic Gradient Descent (SGD)
# Definition: SGD is an optimization algorithm used to minimize a loss function by iteratively updating model parameters.
# Key Feature: It updates parameters using only a single or a few training examples at a time, which can speed up convergence.
# Use Case: Widely used in training machine learning models, especially for large datasets.


