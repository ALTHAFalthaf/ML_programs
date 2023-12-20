from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import plot_tree



breastcancer=load_breast_cancer()
x=breastcancer.data
y=breastcancer.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



 #Initialize Decision Tree classifier with max_depth set to 3
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(x_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=breastcancer.feature_names, class_names=breastcancer.target_names, filled=True)
plt.title("Decision Tree")
plt.show()

v = tree.predict(x_test)
result = accuracy_score(y_test, v)
report = classification_report(y_test, v)

print("Accuracy:", result)
print("\nClassification Report:\n", report)

