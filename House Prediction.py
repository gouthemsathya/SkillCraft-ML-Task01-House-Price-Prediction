import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Select Features and Target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Check Missing Values
print("\nMissing Values:")
print(X.isnull().sum())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("Mean Absolute Error:", mae)
print("Mean Squared Error :", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Display Equation
print("\n===== MODEL COEFFICIENTS =====")
print("Intercept:", model.intercept_)
print("GrLivArea Coefficient:", model.coef_[0])
print("BedroomAbvGr Coefficient:", model.coef_[1])
print("FullBath Coefficient:", model.coef_[2])

# Actual vs Predicted
results = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Predict New House
new_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print("${:,.2f}".format(predicted_price[0]))