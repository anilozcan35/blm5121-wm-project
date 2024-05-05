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
    selected_reg_model = st.selectbox("Fit Regression Model", options)


    #slider for clustering
    st.write("Cluster count - 4 for optima")
    selected_n_cluster = st.slider("Set", min_value=2, max_value=6, value=4)

    clicked = st.button("show")


st.title("Body Performance Data")
st.markdown("Toy model and its applications")

if clicked:
    df = preprocess.preprocess()
    st.table(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Classification Section")
        classification_task = model.ClassificationTask(df)
        report, cm = classification_task.tune_and_predict_classification(selected_reg_model)
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