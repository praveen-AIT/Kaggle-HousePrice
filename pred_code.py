import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

numeric_features = train.select_dtypes(include=[np.number])
target = np.log(train.SalePrice)
corr = numeric_features.corr()

train = train[train['GarageArea'] < 1200]

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

def encode(x):
	if x == 'Partial':
		return 1
	else:
		return 0

train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

lr = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

model = lr.fit(X_train, y_train)

print ("R^2 is: \n", model.score(X_test, y_test))

submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)

final_predictions = np.exp(predictions)

submission['SalePrice'] = final_predictions
submission.to_csv('submission1.csv', index=False)
print(submission.shape)