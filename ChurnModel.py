import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import sys

from typing import Dict, Any, Union


class ChurnModel:
    def __init__(self):
        self.labels = ['SeniorCitizen', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity','OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'PaperlessBilling', 'Churn']
        df = pd.read_csv('./dataset/dataset.csv')
        df.dropna(inplace=True)
        df.set_index('customerID', inplace=True)

        # init new data frame
        labeled_data = df[self.labels]
        labeled_data.insert(loc=0, column='MonthlyCharges', value=df['MonthlyCharges'])

        # save the classes for the production
        json_labels = {}

        self.LEN = {}
        for label in self.labels:
            self.LEN[label] = LabelEncoder()
            self.LEN[label].fit(df[label])
            json_labels[label] = self.LEN[label].classes_
            labeled_data[label] = self.LEN[label].transform(df[label])

        try:
            import cPickle as pickle
        except ImportError:  # python 3.x
            import pickle

        with open('./dataset/labels.p', 'wb') as fp:
            pickle.dump(json_labels, fp, protocol=pickle.HIGHEST_PROTOCOL)

        X = labeled_data.iloc[:, :-1]
        y = labeled_data.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.classifiers = {
            "LogisticRegression": LogisticRegression(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "SVC": SVC()
        }

        # defualt
        self.cls = self.classifiers['LogisticRegression']
        self.cls.fit(self.X_train, self.y_train)

    def train(self, classifier):
        self.cls = self.classifiers[classifier]
        self.cls.fit(self.X_train, self.y_train)

        y_pred = self.cls.predict(self.X_test)
        res = {
            'score': self.cls.score(self.X_test, self.y_test),
            'predicted_score': 'predicted_score',
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }

        return res

    def getLabels(self):
        try:
            import cPickle as pickle
        except ImportError:  # python 3.x
            import pickle

        with open('./dataset/labels.p', 'rb') as fp:
            json_labels = pickle.load(fp)

        return json_labels

    def predict(self, values):
        df = pd.DataFrame(values)
        for label in self.labels:
            if label not in 'Churn':
                le = self.LEN[label]
                le.fit(values[label])
                df[label] = le.transform(df[label])

        return self.cls.predict(df.values[0].reshape(1,-1))[0]
