import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('Salary_Data.csv')

# Preparing the data
x = data['YearsExperience'].values.reshape(-1, 1)  # Reshape should be 'reshape', and column name corrected
y = data['Salary'].values

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Creating a Linear Regression model and fitting it with the training data
model = LinearRegression()
model.fit(x_train, y_train)

# Making predictions on the test set
y_pred = model.predict(x_test)

# Plotting the training data and the regression line
plt.scatter(x_test, y_test, color='red', label='Test data')
plt.plot(x_test, y_pred, color='black', linewidth=2, label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.title('Linear Regression Model')
plt.show()

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2}")