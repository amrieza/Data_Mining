import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

usia_mobil = np.array([5, 4, 6, 5, 5, 5, 6, 6, 2, 7, 7])
harga_mobil = np.array([85, 103, 70, 82, 89, 98, 66, 95, 169, 70, 48])

usia_mobil = usia_mobil.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(usia_mobil, harga_mobil, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Koefisien: ", model.coef_[0])
print("Intercept: ", model.intercept_)

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Usia Mobil (tahun)')
plt.ylabel('Harga Mobil ($100)')
plt.title('Regresi Linear Sederhana')
plt.show()

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
