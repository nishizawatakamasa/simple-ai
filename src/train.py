import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# ダミーデータを作成(y = 2x)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# モデルのトレーニング
model = LinearRegression()
model.fit(X, y)

# モデルを保存
joblib.dump(model, '../models/model.joblib')
