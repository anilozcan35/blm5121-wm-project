import streamlit as st
from src import preprocess
from src import model

def nb_train(model_name = "naive_bayes"):
        st.write("**Training Decision Tree Model**...")
        df = preprocess.get_data()
        df, arg = preprocess.preprocess(pred_mode=False, df=df)
        ct = model.ClassificationTask(dataframe=df, task_type="classification")
        classification_report, figure = ct.tune_and_predict_classification(model_name=model_name)
        st.write("**Training Done, Check Tabs**...")
        return classification_report, figure


def nb_prediction(record_list):
    st.write("**Predicting with Decision Tree Model**...")
    predicted_class = model.ClassificationTask.predict(record_list=record_list, model_name="naive_bayes")
    return predicted_class

