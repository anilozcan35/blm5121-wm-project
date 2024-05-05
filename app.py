import numpy as np
import pandas as pd
import streamlit as st

import src.model as model
import src.preprocess as preprocess

with st.sidebar:
    st.write("sidebar")
    st.button("train")

st.title("Body Performance Data")
st.markdown("Toy model and its applications")
df = preprocess.preprocess()
st.table(df.head())
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Classification Section")
    classification_task = model.ClassificationTask(df)
    report, cm = classification_task.tune_and_predict_classification("naive_bayes")
    st.dataframe(report)
    st.pyplot(cm)

with col2:
    st.markdown("#### Clustering Section")
    clustering_task = model.ClusteringTask(df)
    fig_elbow = clustering_task.show_elbow()
    st.pyplot(fig_elbow)
    fig_clusters = clustering_task.cluster_plots()
    st.pyplot(fig_clusters)

st.markdown("#### Prediction Section")
