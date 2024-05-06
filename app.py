import numpy as np
import pandas as pd
import streamlit as st

import src.model as model
import src.preprocess as preprocess
st.set_page_config(layout="wide")

with st.sidebar:
    st.write("## Model Configuration")

    #regression
    options = ["naive_bayes", "decision_tree", "knn"]
    selected_clas_model = st.selectbox("Fit Regression Model", options)


    #slider for clustering
    st.write("Cluster count - 4 for optima")
    selected_n_cluster = st.slider("Set", min_value=2, max_value=6, value=4)

    show_clicked = st.button("show")

    #prediction
    st.write("# Prediction")
    age = st.text_input(label="Age", value=26.0)
    gender = st.text_input(label="Gender", value="M")
    height_cm = st.text_input(label="HeightCm", value=170)
    weight_kg = st.text_input(label="WeightKg", value=55.8)
    fat = st.text_input(label="Fat", value=15.7)
    diastolic = st.text_input(label="Diastolic", value=77.0)
    systolic = st.text_input(label="Systolic", value=126.0)
    gripForce = st.text_input(label="GripForce", value=36.4)
    forward_cm = st.text_input(label="ForwardCm", value=16.3)
    sit_ups = st.text_input(label="SitUps", value=53.0)
    jump_cm = st.text_input(label="JumpCm", value=29.0)
    classs = "A"
    prediction_record = [float(age),
                         gender,
                         float(height_cm),
                         float(weight_kg),
                         float(fat),
                         float(diastolic),
                         float(systolic),
                         float(gripForce),
                         float(forward_cm),
                         float(sit_ups),
                         float(jump_cm),
                         classs]
    print(prediction_record)
    predict_clicked = st.button("Predict")


st.title("Body Performance Data")
st.markdown("Toy model and its applications")

if show_clicked:
    df, *args = preprocess.preprocess()
    st.table(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Classification Section")
        classification_task = model.ClassificationTask(df)
        report, cm = classification_task.tune_and_predict_classification(selected_clas_model)
        st.dataframe(report)
        st.pyplot(cm)
    with col2:
        st.markdown("#### Clustering Section")
        clustering_task = model.ClusteringTask(df, n_cluster=selected_n_cluster)
        fig_elbow = clustering_task.show_elbow()
        st.pyplot(fig_elbow)
        fig_clusters = clustering_task.cluster_plots()
        st.pyplot(fig_clusters)

if predict_clicked:
    df, *args = preprocess.preprocess()
    st.table(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Classification Section")
        classification_task = model.ClassificationTask(df)
        report, cm = classification_task.tune_and_predict_classification(selected_clas_model)
        pred = classification_task.predict(prediction_record)
        st.dataframe(report)
        st.pyplot(cm)
    with col2:
        st.markdown("#### Clustering Section")
        clustering_task = model.ClusteringTask(df, n_cluster=selected_n_cluster)
        fig_elbow = clustering_task.show_elbow()
        st.pyplot(fig_elbow)
        fig_clusters = clustering_task.cluster_plots()
        st.pyplot(fig_clusters)

    st.markdown("#### Prediction Section")
    st.markdown("### Output")
    st.markdown(pred)
