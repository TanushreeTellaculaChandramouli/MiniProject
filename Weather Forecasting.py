# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("daily_weather.csv")  # Replace with your actual file name
print(df.head())

# Step 3: Preprocess Data
# Drop rows with missing values
df.dropna(inplace=True)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date')

# Add 'next_day_temp' as target
df['next_day_temp'] = df['temperature'].shift(-1)
df.dropna(inplace=True)

# Step 4: Feature Engineering
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month

# Features and Target
features = ['temperature', 'humidity', 'wind_speed', 'day', 'month']
target = 'next_day_temp'

X = df[features]
y = df[target]

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))

# Step 9: Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Samples')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
