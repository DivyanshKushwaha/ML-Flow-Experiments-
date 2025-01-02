import mlflow

import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns


wine = load_wine()
x = wine.data 
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) 

max_depth = 50
n_estimator =150

# mention your experiment below 
mlflow.set_experiment('mlopsexp2')
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator,random_state=42)
    rf.fit(x_train,y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test , y_pred )
    mlflow.log_metric('accuracy ',accuracy )
    mlflow.log_param('max_depth ',max_depth)
    mlflow.log_param('n_estimator ',n_estimator)
    print(accuracy)

    # creating a confusion matrix plot 
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')

    plt.savefig('confusionMatrix.png')
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.log_artifact('confusionMatrix.png')
    mlflow.log_artifact(__file__)


    mlflow.set_tags({'Author':'Divyansh',"Project":"Wine classification"})

    mlflow.sklearn.log_model(rf,"random-forest-classifier")
    print(accuracy)