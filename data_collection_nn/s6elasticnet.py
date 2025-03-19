from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd

test = pd.read_csv('./test_vectors.csv')
train = pd.read_csv('./train_vectors.csv')
X = train.iloc[:,0:train.shape[1]-1]
y = train.iloc[:,-1]
X_test = test.iloc[:,0:test.shape[1]-1]
y_test = test.iloc[:, -1]
## Training an elastic net model
l1_ratio_vals = [x/100.0 for x in range (5, 100, 5)]

model = ElasticNetCV(l1_ratio=l1_ratio_vals)
model.fit(X,  y)
print('Model alpha=', model.alpha_)
print('Model intercept=', model.intercept_)

## Final model
model_final = ElasticNet(l1_ratio=model.l1_ratio_, 
                         alpha=model.alpha_)
model_final.fit(X, y)

# Evaluating on test set
mse = mean_squared_error(y_test, model_final.predict(X_test))
mae = mean_absolute_error(y_test, model_final.predict(X_test))
r2=model_final.score(X_test, y_test)
print("R2:", r2)