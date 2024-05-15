import streamlit as st
from src import preprocess
from src import model


def regression_train():
    st.write("**Training Regression Model**...")
    df, *args = preprocess.preprocess()
    regression_task = model.RegressionTask(df)
    mse, best_params, best_model = regression_task.encode_and_regression(dump=True)
    fig = regression_task.reg_plots()
    st.write("**Training Regression Model Completed. Check Tabs**...")
    st.write("Mean Squared Error =", round(mse,2))
    return mse, best_params, best_model, fig

def regression_predict(record_list):
    st.write("**Predicting Regression Model**...")
    pred = model.RegressionTask.predict(record_list=record_list)
    return pred

