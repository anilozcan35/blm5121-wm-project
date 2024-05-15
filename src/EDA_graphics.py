#%%
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go


def get_data():
    DATA_PATH = "/content/drive/MyDrive/blm5121-wm-project/data/bodyPerformance.csv"
    df = pd.read_csv(DATA_PATH)

    return df

def preprocess_data(data):
    columns_to_rename = {
        'body fat_%': "body_fat_percent",
        'gripForce': "grip_force",
        'sit and bend forward_cm': 'sit_and_bend_forward_cm',
        'sit-ups counts': "sit_ups_counts",
        'broad jump_cm': 'broad_jump_cm',
    }
    data.rename(columns=columns_to_rename, inplace=True)
    data.drop_duplicates(inplace=True)
    for column in data.columns:
        if "cm" in column:
            data[column[:-2] + "m"] = data[column] / 100
            data.drop(column, inplace=True, axis=1)

    data['bmi'] = data.weight_kg / np.power(data.height_m, 2)

    siralama = [
        'age',
        'gender',
        'weight_kg',
        'height_m',
        'bmi',
        'body_fat_percent',
        'grip_force',
        'sit_ups_counts',
        'sit_and_bend_forward_m',
        'broad_jump_m',
        'diastolic',
        'systolic',
        'class',
    ]
    data = data[siralama]
    data.rename(columns={"gender_M": 'gender'}, inplace=True)
    data.gender = data.gender.astype('category')
    data = pd.get_dummies(data=df, columns=['gender'], drop_first=True)
    encode = LabelEncoder()
    data['encoded_class'] = encode.fit_transform(data['class'])
    data.encoded_class = data.encoded_class.astype("category")

    data.drop(columns=['class'], inplace=True)
    return data

def eda_cols(data):
    num_list = data.select_dtypes(exclude=pd.CategoricalDtype).columns.to_list()
    return num_list


def plotting_bar(x:str,group:str, data):
    try:
        fig,ax = plt.subplots(1,2,figsize=(20,10),constrained_layout=True)
        ax = ax.ravel()
        if group == 'gender':
            labels = ['female','male']
            name = group
        else:
            labels = ['A',"B","C","D"]
            name = group
        grouped_agg = data.groupby(group)[x].agg(
            [(f"{x}_mean", 'mean'),(f"{x}_deviation", 'std')]).reset_index()

        my_palette = sns.color_palette("husl",4)
        sns.set_theme(style='whitegrid',rc=rc,palette=my_palette)

        sns.barplot(data=grouped_agg, x=group, y=f'{x}_mean', ax=ax[0], palette = my_palette)
        sns.barplot(data=grouped_agg, x=group, y=f'{x}_deviation', ax=ax[1], palette = my_palette)

        ax[0].set_title(f'Calculated Mean {x}',fontdict=font_title)
        ax[0].set_xlabel(f"{name}", fontdict=font_label)
        ax[0].set_ylabel(f"Calculated Mean {x}", fontdict=font_label)
        ax[0].set_xticklabels(labels,rotation=45,fontsize=20)

        ax[1].set_title(f'Calculated Deviation {x}',fontdict=font_title)
        ax[1].set_xlabel(f"{name}", fontdict=font_label)
        ax[1].set_ylabel(f"Calculated Deviation {x}", fontdict=font_label)
        ax[1].set_xticklabels(labels,rotation=45,fontsize=20)

        fig.suptitle(f"Mean & Deviation Bar Plots"
                    ,fontdict=font_fig,fontweight='bold'
                    ,fontsize=40)
    except KeyError:
            print(f"The wrong Key was passed\nPlease look are the information below\n")
            data.info(memory_usage='deep')

def box_plot(x:str, data):
    objects = data.select_dtypes(include=pd.CategoricalDtype).columns.to_list()
    try:
        fig,ax = plt.subplots(2, figsize=(30,20))
        sns.set_theme(style='whitegrid',rc=rc,palette='bright')
        ax = ax.ravel()

        for index,value in enumerate(objects):
            sns.boxplot(data=data,y=value,x=x,hue=value,ax=ax[index])
            ax[index].set_title(f'Box plot of {x}',fontdict=font_title)
            ax[index].set_xlabel(f"{x}", fontdict=font_label)
            ax[index].set_ylabel(f"{value}", fontdict=font_label)
            fig.suptitle("Box plots",fontdict=font_fig,fontsize=40,fontweight='bold')
    except ValueError:
            print(f"The wrong Value was passed\nPlease look are the information below\n")
            data.info(memory_usage='deep')
    except KeyError:
            print(f"The wrong Key was passed\nPlease look are the information below\n")
            data.info(memory_usage='deep')


rc = {
        'axes.spines.right': True,
        'axes.spines.top': True,
        'font.family': ['sans-serif'],
        'font.sans-serif':
        # 'Arial',
        'DejaVu Sans',
        # 'Liberation Sans',
        # 'Bitstream Vera Sans',
        # 'sans-serif',
        "xtick.bottom":True,
        'axes.edgecolor': 'violet',
        'xtick.color': 'black',
        'figure.facecolor': "snow",
        'grid.color': 'grey',

}


# font definitions
font_label = {'family': 'serif',
        'color':  'darkred',
        'weight': 'semibold',
        'size': 16,
        }

font_title = {'family': 'serif',
        'color':  'black',
        'weight': 'semibold',
        'size': 16,
        }

font_fig = {'family': 'sans',
        'color':  'chocolate',
        # 'weight': 'bold', # doesn't apply to it. Must be specified independently
        # 'fontsize': 30, # doesn't apply to it. Must be specified independently
        }
