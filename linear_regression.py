import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Load Dataset
df = pd.read_csv('housing.csv')
print("Loaded Data:")
print(df)

# Step 2: Explore and Visualize the Data
print("\nData Statistics:")
print(df.describe())

# Plot the data
plt.scatter(df['Size (sqft)'], df['Price ($)'])
plt.title("House Size vs Price")
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()

# Step 3: Train the Linear Regression Model
X = df[['Size (sqft)']]  # Feature must be 2D
y = df['Price ($)']

model = LinearRegression()
model.fit(X, y)

# Step 4: Model Parameters
print(f"\nModel Coefficient (slope): {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")
y_pred = model.predict(X)
print(f"R² Score: {r2_score(y, y_pred):.2f}")

# Step 5: Visualize Regression Line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($)")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Predict Price from User Input
def predict_price():
    try:
        size = float(input("\nEnter house size in square feet: "))
        size_df = pd.DataFrame([[size]], columns=['Size (sqft)'])
        price_pred = model.predict(size_df)[0]
        print(f"Predicted house price: ${price_pred:,.2f}")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

predict_price()
