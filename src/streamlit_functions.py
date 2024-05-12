import os
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport


# @st.cache
def data_preview(file_path: str):
    st.write('**Data Name:**', file_path.split('\\')[-1])
    st.write('**Data Path:**', file_path)
    df_data = pd.read_csv(file_path)
    st.checkbox("Use container width", value=True, key="use_container_width")
    
    # Same as st.write(df)
    # return st.dataframe(df_data)
    return st.dataframe(df_data, use_container_width=st.session_state.use_container_width)


def data_metadata(file_path: str):
    df_data = pd.read_csv(file_path)
    csv = df_data.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(spec=2, gap="small")
    with col1:
        st.link_button("Kaggle Dataset Page", "https://www.kaggle.com/datasets/kukuroo3/body-performance-data")
    with col2:
        # st.link_button("Kaggle Dataset Page", "https://www.kaggle.com/datasets/kukuroo3/body-performance-data")  
        st.download_button(label="Download data as CSV",
                        data=csv,
                        file_name="BodyPerformans.csv",
                        mime="text/csv")

    st.write('**Data Name:**', file_path.split('\\')[-1])
    st.write('**Data Path:**', file_path)
    st.write('**Data Shape:**', str(df_data.shape))
    # st.write('**Data Columns Count:{} & List:{}**'.format(len(df_data.columns), list(df_data.columns)))
    st.write(f'**Data Columns Count:** {len(df_data.columns)}')
    st.write(f'**Data Columns:** {df_data.columns.to_list()}')
    st.write('**Data Types:**', df_data.dtypes)
    st.write('**Data Describe Table:**', df_data.describe())
    # st.write('**Data Info:**', df_data.info(verbose=True))
    st.write('**Data Head Table:**', df_data.head())
    st.write('**Data Tail Table :**', df_data.tail())
    st.write('**Data Sample Table :**', df_data.sample(n=5))


# @st.cache
def data_profiling(file_path: str, report_path: str, minimal: bool = True):
    report_files = ['ProfilingReport.html']
    os.makedirs(report_path, exist_ok=True)
    report_files_exist = all(os.path.isfile(os.path.join(report_path, file)) for file in report_files)
    if not report_files_exist:
        df_data = pd.read_csv(file_path)
        profile = ProfileReport(df=df_data, title="Profiling Report", minimal=minimal)
        profile.to_file(os.path.join(report_path, "ProfilingReport.html"))

    # return profile.to_widgets()
    # return profile.profile.to_notebook_iframe()


def data_profilingA(file_path: str, report_path: str, report_file_name: str, minimal: bool = True):
    print("Report File Name:", report_file_name)
    report_files = []
    report_files.append(report_file_name + ".html")
    os.makedirs(report_path, exist_ok=True)
    report_files_exist = all(os.path.isfile(os.path.join(report_path, file)) for file in report_files)
    if not report_files_exist:
        df_data = pd.read_csv(file_path)
        profile = ProfileReport(df=df_data, title="Profiling Report", minimal=minimal)
        profile.to_file(os.path.join(report_path, report_file_name + ".html"))

