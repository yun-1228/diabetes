############ Import the required Libraries ##################
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Predicting Diabetes Web App")
    st.markdown("Affected by Diabetes or not ?")
    
    st.sidebar.title("Model Selection Panel")
    st.sidebar.markdown("Choose your model and its parameters")
    
    @st.cache_data
    def load_data():# Function to load our dataset
        data = pd.read_csv("diabetes.csv")
        return data
     
    def split(df):# Split the data to 'train and test' sets
        #req_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
        #x = df[req_cols] # Features for our algorithm
        y = df.Outcome
        x = df.drop(columns=['Outcome'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
     
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            
            # Manually plot confusion matrix
            cm = confusion_matrix(y_test, model.predict(x_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap="Blues")
            st.pyplot(plt.gcf())  # Render the current matplotlib figure

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            
            # Compute ROC curve and AUC
            y_prob = model.predict_proba(x_test)[:, 1]  # Assuming binary classification
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            st.pyplot(plt.gcf())  # Render the current matplotlib figure

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            
            # Compute Precision-Recall curve
            y_prob = model.predict_proba(x_test)[:, 1]  # Assuming binary classification
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.figure()
            plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend(loc="lower left")
            st.pyplot(plt.gcf())  # Render the current matplotlib figure
 
    df=load_data() 
    class_names = ['Non-Diabetes', 'Diabetes']
    x_train, x_test, y_train, y_test = split(df)
 
    st.sidebar.subheader("Select your Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision Tree", "Logistic Regression", "Random Forest"))

    if classifier == 'Decision Tree':
        st.sidebar.subheader("Model parameters")
        
        #choose parameters
        criterion= st.sidebar.radio("Criterion(measures the quality of split)", ("gini", "entropy"), key='criterion')
        splitter = st.sidebar.radio("Splitter (How to split at each node?)", ("best", "random"), key='splitter')
        metrics = st.sidebar.multiselect("Select your metrics : ", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
 
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Decision Tree Results")
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",  round(accuracy * 100, 2),"%")
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)
 
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Parameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("Select your metrics?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",  round(accuracy * 100, 2),"%")
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)
 
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap=='True', n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",  round(accuracy * 100, 2),"%")
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Diabetes Raw Dataset")
        st.write(df) 
 
if __name__ == '__main__':
 main()