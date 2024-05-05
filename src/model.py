import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge, SGDClassifier)
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error)
from sklearn.model_selection import (GridSearchCV, StratifiedShuffleSplit,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.cluster import KElbowVisualizer
import src.preprocess as preprocess
import pickle

warnings.filterwarnings("ignore")


class TaskType:
    def __init__(self, task_params, task_type, dataframe):
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.dataframe = dataframe
        if self.task_type == "classification":
            self.task_params = self.task_params = [
                {
                    # DecisionTreeClassifier
                    'min_samples_split': [2, 3, 5],
                    "max_depth": [3, 5, 7],
                },
                {
                    # GaussianNB
                    "var_smoothing": [1e-9, 1e-8, 1e-7]
                },
                {
                    # KNeighborsClassifier
                    'n_neighbors': [3, 4, 5, 6],
                    "weights": ['uniform', 'distance'],
                    "p": [1, 2]
                }]
        else:
            self.task_params =[
                {
                    # Lasso
                    # lower alpha --> higher regularistion (treats underfitting)
                    'alpha': [0.01, 0.1, 0.5, 1],
                    "max_iter": [1000, 2000, 3000, 4000, 10000],
                    "warm_start": [True, False],
                    "tol": [0.001, 0.0001, 0.01, 0.1],
                }]

    def get_model_ready_data(self):
        if self.task_type == "classification":
            X = self.dataframe.drop('encoded_class', axis=1)
            y = self.dataframe['encoded_class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.20)
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            return X_train, X_test, y_train, y_test
        else:
            X = self.dataframe.drop('encoded_class', axis=1)
            y = self.dataframe['encoded_class']
            y = y.replace({0: 25, 1: 50, 2: 75, 3: 100})
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.20)
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            return X_train, X_test, y_train, y_test


class RegressionTask(TaskType):

    def __init__(self, dataframe, task_type="regression", task_params=None, ):
        self.task_type = task_type
        super().__init__(task_params, self.task_type, dataframe)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=.20, random_state=42)


    def encode_and_regression(self, dump = True):
        # Eğitim ve test setlerine ayırma
        X_train, X_test, y_train, y_test = super().get_model_ready_data()

        # Lineer regresyon modeli
        model = Lasso()

        # GridSearchCV kullanarak en iyi modeli bulma
        grid_search = GridSearchCV(model, self.task_params, cv=self.cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # En iyi modeli seçme
        best_model = grid_search.best_estimator_


        # Tahmin yapma
        y_pred = best_model.predict(X_test)

        # Hata ölçümü (örneğin, MSE)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        # En iyi modelin hiperparametrelerini ve performansını yazdırma
        print("Best model parameters:")
        print(best_model.get_params())

        # Eğitim ve test setleri üzerinde tahminleme sonuçları
        print("Prediction on test set:")
        print(y_pred)

        return mse





class ClassificationTask(TaskType):

    def __init__(self, dataframe, task_type="classification", task_params=None):
        self.model_name = None
        self.task_type = task_type
        super().__init__(task_params, self.task_type, dataframe)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=.20, random_state=42)

    def tune_and_predict_classification(self, model_name):
        X_train, X_test, y_train, y_test = super().get_model_ready_data()
        # Model seçimi
        if model_name == 'naive_bayes':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            param_grid = self.task_params[1]
            self.model_name = model_name

        elif model_name == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=42)
            param_grid = self.task_params[0]
            self.model_name = model_name

        elif model_name == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_jobs=-1)
            param_grid = self.task_params[2]
            self.model_name = model_name

        else:
            raise Exception("Geçersiz model adı. Lütfen 'naive_bayes', 'decision_tree' veya 'kneighbors' kullanın.")

        # Grid search ve model uyumu
        grid_search = GridSearchCV(model, param_grid, cv=self.cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # En iyi parametreler ve performans sonuçları
        # print(f"Best parameters for {model_name}:")
        # print(grid_search.best_params_)
        # print(f"Best score: {grid_search.best_score_}")
        # print("\n")
        best_model = grid_search.best_estimator_

        # Tahminleme
        y_pred = best_model.predict(X_test)

        # dump alınması
        with open(str(self.model_name) + 'model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        # confusion matrix create ediliyor.
        fig = plt.figure()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g')

        # Sınıflandırma raporu
        return classification_report(y_test, y_pred, output_dict= True), fig

    def predict(self, record_list):
        raw_df = preprocess.get_data()
        raw_df.loc[len(raw_df)] = record_list
        preprocessed_df_with_record, encoder = preprocess.preprocess(pred_mode=True, df=raw_df)
        with open(str(self.model_name) + 'model.pkl', 'rb') as f:
            model = pickle.load(f)
        record_pred = model.predict(preprocessed_df_with_record[preprocessed_df_with_record.columns[:-1]].tail(1))
        preprocessed_df_with_record.loc[len(preprocessed_df_with_record)]["encoded_class"] = record_pred
        preprocessed_df_with_record["class"] = encoder.inverse_transform(preprocessed_df_with_record["encoded_class"])
        classs = preprocessed_df_with_record.iloc[-1]["class"]

        return classs


class ClusteringTask(TaskType):

    def __init__(self, dataframe, n_cluster=4, task_type="clustering", task_params=None, ):
        self.km = None
        self.scaled_frame = None
        self.task_type = task_type
        self.n_clusters = n_cluster

        super().__init__(task_params, self.task_type, dataframe)

    def clustering_scale_df(self):
        columns_to_scale = self.dataframe.copy()
        to_be_transformed1 = ["weight_kg",
                              "body_fat_percent",
                              "grip_force",
                              "sit_ups_counts",
                              "diastolic",
                              "systolic",
                              'age',
                              'bmi']

        indexes1 = [self.dataframe.columns.to_list().index(column) for column in to_be_transformed1]
        standard_scaler = StandardScaler()
        column_transformer = ColumnTransformer(
            transformers=[
                (
                    'standard_scalering', standard_scaler, indexes1
                )], remainder="passthrough")
        scaled = column_transformer.fit_transform(columns_to_scale)
        self.scaled_frame = pd.DataFrame(scaled, columns=self.dataframe.columns.to_list())
        #print(self.scaled_frame)

        return self.scaled_frame

    def show_elbow(self):
        fig = plt.figure()
        km = KMeans(random_state=42)
        visualiser = KElbowVisualizer(km, k=(2, 10), metric='distortion')
        visualiser.fit(self.clustering_scale_df()) # dataframe scale ediliyor ve fit ediliyor.
        visualiser.show()

        return fig

    def km_optima(self):
        km = KMeans(n_clusters=self.n_clusters, random_state=42)
        km.fit(self.scaled_frame)
        # print(km.labels_, "\n")
        # print(km.cluster_centers_, "\n")
        self.km = km

    def cluster_plots(self, x_index=0, y_index=1, rc=None):
        fig = plt.figure()
        self.km_optima()
        sns.set_theme(rc=rc, style='whitegrid', palette='bright')
        x_column_name = self.scaled_frame.columns[x_index]
        y_column_name = self.scaled_frame.columns[y_index]

        sns.scatterplot(x=self.scaled_frame.iloc[:, x_index], y=self.scaled_frame.iloc[:, y_index], hue=self.km.labels_,
                        palette='bright')
        plt.scatter(self.km.cluster_centers_[:, x_index], self.km.cluster_centers_[:, y_index], marker='X', s=80,
                    label="centroids", color='red')
        plt.xlabel(f"{x_column_name}")
        plt.ylabel(f"{y_column_name}")
        plt.title("Cluster plots")
        plt.legend()
        return fig

df = preprocess.preprocess()
classifcation_task = ClassificationTask(df)
prediction = classifcation_task.tune_and_predict_classification("naive_bayes")
print(prediction)

DATA_PATH = "data/bodyPerformance.csv"
raw_df = pd.read_csv(DATA_PATH)

raw_df["class"].value_counts()

df.iloc[1]
df.tail(1)

age = 28
gender = "M"
height_cm = 170
weight_kg = 55.8
fat = 15.7
diastolic = 77.0
systolic = 126.0
gripForce = 36.4
forward_cm = 16.3
sit_ups = 53.0
jump_cm = 229.0
classs = "A" #rastgele bir label


ss = [age, gender, height_cm, weight_kg, fat, diastolic, systolic, gripForce, forward_cm, sit_ups, jump_cm, classs]
raw_df.loc[len(raw_df)] = ss



raw_df["class"]
if __name__ == '__main__':

    df = preprocess.preprocess()
    classifcation_task = ClassificationTask(df)
    prediction = classifcation_task.tune_and_predict_classification("decision_tree")
    pred = classifcation_task.predict(ss)
    print(pred)

    df = preprocess.preprocess()
    clusteringtask = ClusteringTask(df)
    clusteringtask.show_elbow()
    clusteringtask.cluster_plots()

    df = preprocess.preprocess()
    regressiontask = RegressionTask(df)
    regressiontask.encode_and_regression()
