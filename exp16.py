from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score ,classification_report
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
print(clf.predict(x_test))
v=clf.predict(x_test)
result=accuracy_score(y_test,v)
print("accuracy:",result)
result=classification_report(y_test,v)
print("classification_report:",result)

new_data_point = [[5.1, 3.5, 1.4, 0.2]]
new_prediction = clf.predict(new_data_point)
predicted_category=iris.target_names[new_prediction[0]]
print("Predicted class for new data point:",predicted_category )