from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)
print("DevOps CI Pipeline Working Again")
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)