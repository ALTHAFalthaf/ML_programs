from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize Decision Tree classifier with max_depth set to 3
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(x_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree")
plt.show()

# Make predictions and evaluate the model
v = tree.predict(x_test)
result = accuracy_score(y_test, v)
report = classification_report(y_test, v)

print("Accuracy:", result)
print("\nClassification Report:\n", report)
