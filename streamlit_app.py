import os
import warnings

# import threading
# import time
# from datetime import datetime
# import pandas as pd

# import src.model as model
# import src.preprocess as preprocess
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.functions import download_dataset_from_kaggle
from src.streamlit_functions import (data_metadata, data_preview,
                                     data_profiling, data_profilingA)
from src.models import dt_algoritm, knn_algoritm, nb_algorithm, k_means

from src.models.dt_algoritm import dt_train, dt_prediction

# from st_pages import show_pages_from_config

warnings.filterwarnings("ignore")


def home():
    """Home Page of Streamlit UI"""

    st.title('Web Mining Project Title', anchor='top', help='Web Mining Project Help')
    st.header('Web Mining Project Header')
    st.subheader('Bu uygulama Web Mining (BLM-5121) Projesi kapsamında ML Algoritmaları için geliştirilmiştir.')
    st.markdown('**1. Project Proposal:** Proje önerisi ve proje hakkında bilgi alabilirsiniz.')
    st.markdown('**2. Project System Design:** Proje aşamaları ve sistem tasarımı hakkında bilgi alabilirsiniz.')
    st.markdown('**3. Dataset Info:** Veri seti önizlemesi yapabilirsiniz. Veri seti hakkında bilgi alabilirsiniz.')
    st.markdown(
        '**4. Multi Class Classification:** Çoklu sınıflandırma uygulamasıdır. Veri seti üzerinde çoklu sınıflandırma yapabilirsiniz.')
    st.markdown('**5. Regression:** Regresyon uygulamasıdır. Veri seti üzerinde regresyon analizi yapabilirsiniz.')
    st.markdown('**6. Clustering:** Kümeleme uygulamasıdır. Veri seti üzerinde kümeleme analizi yapabilirsiniz.')
    st.markdown('**7. App Info. & Credits:** Bu projede kullanılan Framework ve Libraryleri içermektedir.')


def proposal():
    """Project Proposal Page"""

    with open(file="ProjectProposal.md", encoding="utf8") as p:
        st.markdown(p.read())


def pipeline():
    """Project System Design Page"""
    st.title('Project System Design Title')
    st.header('Project System Design Header')
    st.subheader('Proje Sistemi Tasarımı: Proje aşamaları ve sistem tasarımı hakkında bilgi alabilirsiniz.')
    st.image(image="./pipeline/SystemDesign.jpg",
             caption="Project System Design",
             width=200,
             use_column_width="auto"
             )


def data():
    """Dataset Information Page"""

    st.title('Dataset Information Title')
    st.header('Dataset Information Header')
    st.subheader('Veri seti önizlemesi yapabilirsiniz. Veri seti hakkında bilgi alabilirsiniz.')
    tab1, tab2, tab3, tab4 = st.tabs(["Meta Data", "Preview", "Profile(Raw Data)", "Profile(Preprocess Data)"])
    with tab1:
        st.image(
            image="https://storage.googleapis.com/kaggle-datasets-images/1732554/2832282/1be2ae7e0f1bc3983e65c76bfe3a436e/dataset-cover.jpg?t=2021-11-20-09-31-54",
            caption="Body Performance Dataset from Kaggle",
            width=200,
            use_column_width="auto"
        )
        st.title('Meta Data')
        st.header("Meta Data")
        data_metadata(file_path=DATA_FILE)
        # st.page_link(page="http://www.google.com", label="Dataset Url: Kaggle", icon="🌎")
    with tab2:
        st.title('Data Preview')
        st.header("Data Preview")
        data_preview(file_path=DATA_FILE)
    with tab3:
        st.title('Raw Data Profiling')
        st.header("Raw Data Profiling")
        with open(file="data/profiling/RawDataProfilingReport.html", encoding="utf8") as p:
            components.html(p.read(), height=4096, width=2160, scrolling=True)
    with tab4:
        st.title('Preprocess Data Profiling')
        st.header("Preprocess Data Profiling")
        with open(file="data/profiling/PreprocessDataProfilingReport.html", encoding="utf8") as p:
            components.html(p.read(), height=4096, width=2160, scrolling=True)


def get_prediction_records(key_start=0):
    age = st.text_input(label="Age", value=26.0, key=key_start)
    gender = st.text_input(label="Gender", value="M", key=key_start + 1)
    height_cm = st.text_input(label="HeightCm", value=170, key=key_start + 2)
    weight_kg = st.text_input(label="WeightKg", value=55.8, key=key_start + 3)
    fat = st.text_input(label="Fat", value=15.7, key=key_start + 4)
    diastolic = st.text_input(label="Diastolic", value=77.0, key=key_start + 5)
    systolic = st.text_input(label="Systolic", value=126.0, key=key_start + 6)
    gripForce = st.text_input(label="GripForce", value=36.4, key=key_start + 7)
    forward_cm = st.text_input(label="ForwardCm", value=16.3, key=key_start + 8)
    sit_ups = st.text_input(label="SitUps", value=53.0, key=key_start + 9)
    jump_cm = st.text_input(label="JumpCm", value=29.0, key=key_start + 10)
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
    return prediction_record


def classification():
    """Multi Class Classification Page"""

    st.title('Multi Class Classification Title')
    st.header('Multi Class Classification Algorithms Header')
    st.subheader('Çoklu sınıflandırma uygulamasıdır. Veri seti üzerinde çoklu sınıflandırma yapabilirsiniz.')
    # option = st.selectbox("Multi Class Classification Alogritmaları?",
    #                       ("Decision Tree", "KNN", "LightGBM"),
    #                       index=None,
    #                       placeholder="Model seçiniz...",
    #                       )
    # st.write("Model Seçimi:", option)

    tab1, tab2, tab3 = st.tabs(["Decision Tree (DT)", "K Nearest Neighbor (KNN)", "Naive Bayes"])
    with tab1:
        st.header("Decision Tree")
        st.write('Decision Tree işlemi yapılacak.')

        tab1_1, tab1_2, tab1_3, tab1_4 = st.tabs(
            ["Training Component Component", "Model Charts", "Prediction Component", "Other"])
        with tab1_1:
            st.header("Decision Tree Training Component")
            st.write('Decision Tree Training Process Will Be Done.')
            dt_train_button = st.button("Train DT")

            if dt_train_button:
                classification_report, cm = dt_algoritm.dt_train(model_name="decision_tree")

                with tab1_2:
                    st.header("Decision Tree Model Charts")
                    st.pyplot(cm)

                with tab1_4:
                    st.header("Decision Tree Other")
                    st.dataframe(classification_report)

            with tab1_3:
                st.header("Decision Tree Prediction")
                st.write('Decision Tree Prediction Will Be Done.')
                prediction_record = get_prediction_records(key_start=0)
                print(prediction_record)
                predict_clicked = st.button("Predict", key = 100)

                if predict_clicked:
                    pred = dt_algoritm.dt_prediction(prediction_record)
                    st.header("Prediction")
                    st.write(pred)

                    with tab1_1:
                        classification_report, cm = dt_algoritm.dt_train(model_name="decision_tree")

                    with tab1_2:
                        st.header("Decision Tree Model Charts")
                        st.pyplot(cm)

                    with tab1_4:
                        st.header("Decision Tree Other")
                        st.dataframe(classification_report)

    with tab2:
        st.header("K Nearest Neighbor")
        st.write("K Nearest Neighbor Process Will Be Done.")

        tab2_1, tab2_2, tab2_3, tab2_4 = st.tabs(
            ["Training Component Component", "Model Charts", "Prediction Component", "Other"])
        with tab2_1:
            st.header("K Nearest Neighbor Training Component")
            st.write('K Nearest Neighbor Process Will Be Done.')
            knn_train_button = st.button("Train KNN")

            if knn_train_button:
                classification_report, cm = knn_algoritm.knn_train(model_name="knn")

                with tab2_2:
                    st.header("K Nearest Neighbor Model Charts")
                    st.pyplot(cm)

                with tab2_4:
                    st.header("K Nearest Neighbor Other")
                    st.dataframe(classification_report)

            with tab2_3:
                st.header("K Nearest Neighbor Prediction")
                st.write("K Nearest Neighbor Process Will Be Done.")
                prediction_record = get_prediction_records(key_start=11)
                print(prediction_record)
                knn_predict_clicked = st.button("Predict", key=101)

                if knn_predict_clicked:
                    pred = knn_algoritm.knn_prediction(prediction_record)
                    st.header("Prediction")
                    st.write(pred)

                    with tab2_1:
                        classification_report, cm = knn_algoritm.knn_train(model_name="knn")

                    with tab2_2:
                        st.header("K Nearest Neighbor Model Charts")
                        st.pyplot(cm)

                    with tab2_4:
                        st.header("Other")
                        st.dataframe(classification_report)

    with tab3:
        st.header("Naive Bayes")
        st.write("Naive Bayes Process Will Be Done.")

        tab3_1, tab3_2, tab3_3, tab3_4 = st.tabs(
            ["Training Component Component", "Model Charts", "Prediction Component", "Other"])
        with tab3_1:
            st.header("Naive Bayes Training Component")
            st.write('Naive Bayes Training Process Will Be Done.')
            nb_train_button = st.button("Train NB", key=103)

            if nb_train_button:
                classification_report, cm = nb_algorithm.nb_train(model_name="naive_bayes")

                with tab3_2:
                    st.header("Naive Bayes Model Charts")
                    st.pyplot(cm)

                with tab3_4:
                    st.header("Other")
                    st.dataframe(classification_report)

            with tab3_3:
                st.header("Naive Bayes Prediction")
                st.write("Naive Bayes Prediction Process Will Be Done.")
                prediction_record = get_prediction_records(key_start=22)
                print(prediction_record)
                nb_predict_clicked = st.button("Predict", key=104)

                if nb_predict_clicked:
                    pred = nb_algorithm.nb_prediction(prediction_record)
                    st.header("Prediction")
                    st.write(pred)

                    with tab3_1:
                        classification_report, cm = nb_algorithm.nb_train(model_name="naive_bayes")

                    with tab3_2:
                        st.header("Naive Bayes Model Charts")
                        st.pyplot(cm)

                    with tab3_4:
                        st.header("Other")
                        st.dataframe(classification_report)



def regression():
    """Regression Page"""

    st.title('Regression Title')
    st.header('Regression Algorithms Header')
    st.subheader('Regresyon uygulamasıdır. Veri seti üzerinde regresyon analizi yapabilirsiniz.')
    tab1, tab2, tab3 = st.tabs(["Training Component", "Model Charts", "Prediction Component"])

    with tab1:
        st.header("Training Component")
        st.write('Training Component işlemi yapılacak.')

    with tab2:
        st.header("Model Charts")
        st.write('Model Charts işlemi yapılacak.')
        tab2_1, tab2_2, tab2_3 = st.tabs(["Loss Model Charts", "Accuracy Model Charts", "Other Model Charts"])
        with tab2_1:
            st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        with tab2_2:
            st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        with tab2_3:
            st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with tab3:
        st.header("Prediction Component")
        st.write('Prediction Component işlemi yapılacak.')


def clustering():
    """Clustering Page"""

    st.title('Clustering Section')
    st.header('K Means Algorithm')
    st.subheader('Clustering jobs could be executed in this section.')
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Selection Training Param", "Elbow Graph", "Clusters", "Prediction Section"])

    with tab1:
        st.header("Training Component")
        st.write('Selection Training Param')
        selected_n_cluster = st.slider("Set K", min_value=2, max_value=6, value=4)
        show_clicked = st.button("Train K-means", key=106)
        if show_clicked:
            fig_elbow, fig_clusters = k_means.k_means_train(selected_n_cluster=selected_n_cluster)
            with tab2:
                st.subheader("Elbow Graph")
                st.write('K Params Optimal for 4')
                st.pyplot(fig_elbow)
            with tab3:
                st.subheader("Clusters")
                st.pyplot(fig_clusters)

        with tab4:
            st.header("K-Means Prediction")
            st.write("K-Means Prediction Process Will Be Done.")
            prediction_record = get_prediction_records(key_start=33)
            print(prediction_record)
            cluster_predict = st.button("Predict", key=105)

            if cluster_predict:
                fig = k_means.k_means_predict(prediction_record, selected_n_cluster)
                st.header("Prediction of Point")
                st.pyplot(fig)

                fig_elbow, fig_clusters = k_means.k_means_train(selected_n_cluster=selected_n_cluster)
                with tab2:
                    st.subheader("Elbow Graph")
                    st.write('K Params Optimal for 4')
                    st.pyplot(fig_elbow)
                with tab3:
                    st.subheader("Clusters")
                    st.pyplot(fig_clusters)
                    # Buralardaki kod tekrarnı nasıl çözülebilir.



def credits():
    """App Info. & Credits Page"""

    st.title('App Info. & Credits Title')
    st.header('App Info. & Credits Header')
    st.subheader('App Info. & Credits: Bu projede kullanılan Framework ve Libraryleri içermektedir.')
    st.write('App Info. & Credits: Bu projede kullanılan Framework ve Libraryleri içermektedir.')

    st.markdown('**Programming Language:** Python 3.12')
    st.markdown('**Libraries & Frameworks:** Pandas, Scikit-learn, Numpy, Matplotlib, Seaborn, Plotly')
    st.markdown('**UI:** Streamlit')
    st.markdown('**Dev. Tools:** Docker & Git')
    st.markdown('**Dash Url:** [StreamLit App](https://web-mining-project.streamlit.app/)')
    st.markdown('**Developed by:** Metin Uslu & Anıl Özcan')
    # st.page_link(page="http://www.google.com", label="Google", icon="🌎")
    # st.page_link("your_app.py", label="Home", icon="🏠")
    # st.page_link("pages/page_1.py", label="Page 1", icon="1️⃣")
    # st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)


def menu(user_name=None, user_password=None):
    """Streamlit UI Menu
    Params:
        user_name: str 
        user_password: str
    """

    st.sidebar.title('Web Mining Project')
    menu = {
        'Giriş': home,
        'Project Proposal': proposal,
        'Project System Design': pipeline,
        'Dataset Info': data,
        'Multi Class Classification Algorithms': classification,
        'Regression Algorithms': regression,
        'Clustering Algorithms': clustering,
        'App. Info. & Credits': credits
    }

    if st.session_state.get('login_success'):
        choice = st.sidebar.radio('Applications', list(menu.keys()))
        menu[choice]()
    else:
        with st.sidebar:
            with st.form(key='login_form'):
                st.title('Loging Page')
                username = st.text_input('User Name')
                password = st.text_input('Password', type='password')
                if st.form_submit_button('Login'):
                    if username == user_name and password == user_password:
                        st.session_state['login_success'] = True
                        st.success('Giriş başarılı, yönlendiriliyorsunuz...')
                        st.experimental_rerun()
                    else:
                        st.error('Kullanıcı adı veya şifre yanlış.')
                        st.session_state['login_success'] = False
    # show_pages_from_config()


if __name__ == "__main__":
    # Set Constants
    ROOT_PATH = os.getcwd()
    CFG_PATH = os.path.join(ROOT_PATH, 'cfg')
    ENV = os.path.join(CFG_PATH, '.env')
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'preprocessed')
    PROFILLING_PATH = os.path.join(DATA_PATH, 'profiling')

    DATA_FILE = os.path.join(RAW_DATA_PATH, 'bodyPerformance.csv')

    # Load Environment Variables
    load_dotenv(dotenv_path=ENV, encoding='utf-8', verbose=False)

    # Get Constants
    #USER_NAME = os.environ.get("USER_NAME")
    #USER_PASSWORD = os.environ.get("USER_PASSWORD")

    st.set_page_config(
        page_title="Web Mining Project UI ",
        page_icon=":gem:",
        layout="wide",
        # layout="centered",        
        initial_sidebar_state="expanded",
        # initial_sidebar_state="auto",
        # menu_items=None,
        menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
                    'Report a bug': "https://www.extremelycoolapp.com/bug",
                    'About': "# This is a header. This is an *extremely* cool app!"
                    }
    )

    download_dataset_from_kaggle(user_name="kukuroo3", dataset_name="body-performance-data", path=RAW_DATA_PATH)
    # data_profiling(file_path=DATA_FILE, report_path=PROFILLING_PATH, minimal=False)
    data_profilingA(file_path=DATA_FILE, report_path=PROFILLING_PATH, minimal=False,
                    report_file_name="RawDataProfilingReport")
    data_profilingA(file_path=DATA_FILE, report_path=PROFILLING_PATH, minimal=False,
                    report_file_name="PreprocessDataProfilingReport")
    menu(user_name='local', user_password='local')  # login için değiştirildi.
