import streamlit as st
from src import preprocess
from src import  model

# Şimdilik params none bırakıyorum zaman kalırsa impelemte edilir.
def knn_train(model_name = "knn", params = None):
        st.write("**Training KNN Model**...")
        df = preprocess.get_data()
        df, arg = preprocess.preprocess(pred_mode=False, df=df)
        ct = model.ClassificationTask(dataframe=df, task_type="classification")
        classification_report, figure = ct.tune_and_predict_classification(model_name=model_name)
        st.write("**Training Done, Check Tabs**...")
        return classification_report, figure

def knn_prediction(record_list):
    st.write("**Predicting with KNN Model**...")
    predicted_class = model.ClassificationTask.predict(record_list=record_list, model_name="knn")
    return predicted_class

