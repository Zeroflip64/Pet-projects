import streamlit as st
import pandas as pd

df=pd.read_csv('/content/drive/MyDrive/учеба/marketing_campaign.csv',sep='\t')
my_colors=['#730080','#00ab66','#636363','#779f73']



st.title("DataFrame of this company")
st.table(df.head())

st.write("Let's create a correlation matrix between the features to determine which features are important")

