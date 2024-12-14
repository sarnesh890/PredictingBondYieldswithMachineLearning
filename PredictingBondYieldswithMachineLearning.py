import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset with historical bond data (e.g., bond prices, maturity, coupon rate, etc.)
data = pd.read_csv("bond_data.csv")

# Define the features (e.g., coupon rate, maturity, interest rates)
features = data[['CouponRate', 'MaturityYears', 'CurrentPrice', 'InterestRate']]
target = data['Yield']  # Yield is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict yields on the test set
y_pred = model.predict(X_test)

# Calculate and print the Mean Absolute Error (MAE) of the predictions
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error of yield predictions:", mae)
