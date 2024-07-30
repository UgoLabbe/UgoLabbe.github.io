import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate example data: Fertilizer amount (in grams) and corresponding Plant Growth (in cm)
np.random.seed(0)
fertilizer = np.random.uniform(0, 100, 100).reshape(-1, 1)  # Generate 100 data points for fertilizer amount
plant_growth = 0.1 * fertilizer ** 2 - 0.5 * fertilizer + np.random.normal(0, 30, size=(100, 1))  # Quadratic relationship with more noise

# Add some outliers
outliers_fertilizer = np.array([20, 30, 70, 80]).reshape(-1, 1)
outliers_growth = np.array([150, 50, 120, 20]).reshape(-1, 1)

# Combine the data points and outliers
fertilizer = np.vstack((fertilizer, outliers_fertilizer))
plant_growth = np.vstack((plant_growth, outliers_growth))

# Fit a polynomial regression model
poly = PolynomialFeatures(degree=2)
fertilizer_poly = poly.fit_transform(fertilizer)
model = LinearRegression()
model.fit(fertilizer_poly, plant_growth)
fertilizer_new = np.linspace(0, 100, 100).reshape(-1, 1)
fertilizer_new_poly = poly.transform(fertilizer_new)
plant_growth_predict = model.predict(fertilizer_new_poly)

# Plot the data and the regression curve
plt.figure(figsize=(10, 6))
plt.scatter(fertilizer, plant_growth, color='blue', label='Data points')
plt.plot(fertilizer_new, plant_growth_predict, color='red', linewidth=2, label='Regression curve')
plt.xlabel('Fertilizer Amount (grams)')
plt.ylabel('Plant Growth (cm)')
plt.title('Plant Growth Prediction Using Polynomial Regression with Noise and Outliers')
plt.legend()
plt.grid(True)
plt.savefig('fertilizer_plant_growth_regression.png')
plt.show()
