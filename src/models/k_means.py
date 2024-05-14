import streamlit as st
from src import preprocess
from src import model


def k_means_train(selected_n_cluster):
    st.write("**Training k-Means Model**...")
    df, *args = preprocess.preprocess()
    clustering_task = model.ClusteringTask(df, n_cluster=selected_n_cluster)
    fig_elbow = clustering_task.show_elbow()
    fig_clusters = clustering_task.cluster_plots()
    st.write("**Training k-Means Model Completed. Check Tabs**...")
    return fig_elbow, fig_clusters

def k_means_predict(record_list, selected_n_cluster):
    st.write("**Predicting k-Means Model**...")
    fig = model.ClusteringTask.predict(record_list=record_list, n_clusters=selected_n_cluster)
    return fig


def knn_prediction(record_list):
    st.write("**Predicting with KNN Model**...")
    predicted_class = model.ClassificationTask.predict(record_list=record_list, model_name="knn")
    return predicted_class