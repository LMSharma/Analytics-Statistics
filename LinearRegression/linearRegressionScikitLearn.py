############# Linear Regression Using Scikit Learn ###################

from numpy import genfromtxt,array
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import linear_model

regressor = linear_model.LinearRegression()



points = genfromtxt('data.csv',delimiter =",")
    
x = [[points[i,0]] for i in range(len(points))]
y = [[points[i,1]] for i in range(len(points))]

#print(x)
#print(y)

#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state = 4)

#print(len(x_train))


#regressor.fit(x_train,y_train)
regressor.fit(x,y)


m = regressor.coef_[0][0]
b = regressor.intercept_[0]

y_min = m*min(x)[0]+b
y_max = m*max(x)[0]+b
print("value of m and b are : ",m,b)
plt.scatter(x,y,color='green')
plt.plot([min(x) , max(x)] , [y_min ,y_max] , 'r')
plt.xlabel('# hr studied')
plt.ylabel('marks')
plt.show()

#a = regressor.predict(x_test)
#print(a)
