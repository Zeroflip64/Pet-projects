
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D #3d оси
from yellowbrick.cluster import KElbowVisualizer#выбор кролчества кластеров
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,RobustScaler,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Project: Clustering of store customers to identify the best offers within the store chain.")
st.write('We have customer data of the grocery store chain.We will divide the project into several tasks:')


st.text('Task 1: It is necessary to identify the model that will best be able to divide buyers.')
st.text('Task 2: To analyze the resulting groups.')
st.text('Task 3: Draw conclusions and make suggestions.')

df = pd.read_csv('https://github.com/Zeroflip64/Pet-projects/raw/main/marketing_campaign.csv',sep='\t')
my_colors=['#730080','#00ab66','#636363','#779f73']



st.title("DataFrame of this company")
st.table(df.head())

st.write("Let's create a correlation matrix between the features to determine which features are important")

