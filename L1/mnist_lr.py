from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

digits = load_digits()
data = digits.data

print(data.shape)

print(digits.images[0])

print(digits.target[0])

plt.gray()
plt.imshow(digits.images[0])
plt.show()


train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)


ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)


#lr = LogisticRegression()
lr = DecisionTreeRegressor()
lr.fit(train_ss_x, train_y)
predict_y=lr.predict(test_ss_x)
print('LR accurate rate: %0.4lf' % accuracy_score(predict_y, test_y))
