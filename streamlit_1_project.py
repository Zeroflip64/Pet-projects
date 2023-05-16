import pandas as pd
from sklearn.decomposition import PCA
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

df=pd.read_csv('https://github.com/Zeroflip64/Pet-projects/blob/main/marketing_campaign.csv',sep='\t')
my_colors=['#730080','#00ab66','#636363','#779f73']



st.title("DataFrame of this company")
st.table(df.head())

st.write("Let's create a correlation matrix between the features to determine which features are important")


matrix_corr=df.corr()
fig, ax = plt.subplots(figsize=(20, 10))
go.Figure(data=go.Heatmap(z=matrix_corr))
sns.heatmap(matrix_corr, annot=True, ax=ax)
st.pyplot(fig)
plt.close()  # Close the first graph

st.write("Let's check the numerical signs for outliers and we use Z-score to determine outliers")

for i in [i for i in df.columns if df[i].dtype=='int']:
  data=df[i]
  z_scores = stats.zscore(data)
  threshold = 3
  outliers = df[np.abs(z_scores) > threshold]
  df.loc[outliers.index, i] = None
df.isna().sum().plot(kind='barh')
st.pyplot()

st.text('As we can see from the heat map and graph, we have columns that do not carry significance and abnormal data that we will delete')

for i in ['NumWebPurchases', 'NumCatalogPurchases','NumWebVisitsMonth']:
  df=df.loc[df[i]<15]
df=df.drop(['Complain','AcceptedCmp2','Z_CostContact','Z_Revenue','AcceptedCmp3','NumDealsPurchases','Recency','ID','Dt_Customer','Response'],axis=1)
df=df.loc[df['Income']<100000]#We also see values that are knocked out of the total mass in the amount of profit
df=df.loc[df['Year_Birth']>1945]#We can see anomalies in the column with the date of birth, and therefore we will remove everyone older than 1945
df=df.loc[df['NumWebPurchases']+df['NumCatalogPurchases']+df['NumStorePurchases']!=0]#Removed those customers who did not make purchases in stores
df['Year_Birth']=df['Year_Birth'].astype('object')#Change of types 
columns_object=[i for i in  df.columns if df[i].dtypes!='int' and i!='Income' ]
df=df.dropna()

st.write('Creating new features')
st.text("Let's combine the family statuses.")
st.text("Let's combine the signs of children and adolescents in the family.")
st.text("Let's identify the percentage of purchases made directly in the store.")
st.text("We will add new signs in the form of an average basket and customer activity.")

import numpy as np
bad=['Absurd','Alone','YOLO']
df=df.query('Marital_Status not in @bad')

def maried_status(data):
  
  if data['Marital_Status']in ['Married','Together']:
    return 'Married'
  elif data['Marital_Status'] in ['Divorced','Widow']:
    return 'Divorced'
  else:
    return data['Marital_Status']

def offline(data):
  online=data['NumWebPurchases']+data['NumCatalogPurchases']
  offline=data['NumStorePurchases']
  if online==0:
    return 1
  elif offline==0:
    return 0
  else:
    return np.round(offline/(online+offline),2)

df['Marital_Status']=df.apply(maried_status,axis=1)

df['Childs']=df['Kidhome']+df['Teenhome']

df['count_of_purchases']=df['NumWebPurchases']+df['NumCatalogPurchases']+df['NumStorePurchases']

df['Basket']=np.round(df['Income']/(df['NumWebPurchases']+df['NumCatalogPurchases']+df['NumStorePurchases']),2)

df['procent_offline']=df.apply(offline,axis=1)

df['is_active']=((df['AcceptedCmp4']+df['AcceptedCmp5']+df['AcceptedCmp1'])>=2).map(int)

df=df.drop(['Kidhome','Teenhome','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1'],axis=1)

st.write('Our datframe with new columns')
st.table(df.head())

st.write('')
columns_num=[i for i in  df.columns if df[i].dtypes!='object' and i not in ['is_active','Response']]

preprocesing=make_column_transformer((OrdinalEncoder(),['Education','Marital_Status']),
                                     (StandardScaler(),columns_num),remainder='passthrough')
important=df[[i for i in df.columns][:4]]


df_encoder=pd.DataFrame(preprocesing.fit_transform(df),columns=df.columns)
df_encoder.dropna(inplace=True)

st.write('Due to the fact that we have quite a lot of features and there is also a correlation between them, we will use the PCA algorithm in order to reduce the dimension')

pca=PCA(n_components=3)
df_pca=pd.DataFrame(pca.fit_transform(df_encoder),columns=['list_c1','list_c2','list_c3'])

st.write('Transformed data in space')

x =df_pca["list_c1"]
y =df_pca["list_c2"]
z =df_pca["list_c3"]
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="cyan", marker="o" )
st.pyplot(fig)


st.title('Task 1: It is necessary to identify the model that will best be able to divide buyers.')


class claster:
  def __init__(self,data):
    self.data=data

  def preprocesing(self,metod_of_object,metod_of_num):
    columns_num=[i for i in  self.data.columns if self.data[i].dtypes!='object' and i not in ['is_active','Response']]

    preprocesing=make_column_transformer((metod_of_object,['Education','Marital_Status']),
                                     (metod_of_num,columns_num),remainder='passthrough')
    
    self.df_encoder=pd.DataFrame(preprocesing.fit_transform(self.data),columns=self.data.columns)
    print('+')
    pca=PCA(n_components=3)
    self.df_pca=pd.DataFrame(pca.fit_transform(self.df_encoder),columns=['list_c1','list_c2','list_c3'])
    return self.df_pca

  def detected(self,algoritm,data):
    self.algoritm=algoritm
    Elbow_M = KElbowVisualizer(self.algoritm, k=10)
    Elbow_M.fit(data)
    Elbow_M.show()

  def work(self,model,data,num,score):
    models=model.fit(data)
    data['labels']=models.labels_
    silhouette_score=score[0](self.df_pca, models.labels_)
    davies_bouldin_score=score[1](self.df_pca, models.labels_)
    calinski_harabasz_score=score[2](self.df_pca, models.labels_)
    
    figer= go.Figure(data=[go.Scatter3d(x=data['list_c1'],y=data['list_c2'],
    z=data['list_c3'],mode='markers',marker=dict(
            size=4,
            color=data['labels'], 
            opacity=0.8))])
          
    st.plotly_chart(figer)
    plt.show()
    st.write(pd.DataFrame({'Score':[silhouette_score,davies_bouldin_score,calinski_harabasz_score]},index=['silhouette_score','davies_bouldin_score','calinski_harabasz_score']))
    return model.labels_
    

st.write('I decided to use two algorithms these are K means and Agglomerative (since these are algorithms of different principles, it will be interesting to see which of them will show the best indicator)')
st.text('We use three metrics :')
st.text('1) Silhouette score: The silhouette score measures how well each data point is separated from its own cluster compared to other clusters. It ranges from -1 to 1, with higher values indicating better cluster separation.')
st.text('2) Calinski-Harabasz index: The Calinski-Harabasz index measures the ratio of the between-cluster variance to the within-cluster variance. Higher values indicate better cluster separation.')
st.text('3) Davies-Bouldin index: The Davies-Bouldin index measures the average similarity between each cluster and its most similar cluster, and the average distance between each cluster and its least similar cluster. Lower values indicate better cluster separation.')

st.write('The Best model')
st.write('The best average of the three metrics was shown by the K means algorithm for 4 clusters in the future we will use it.')
data=claster(df)
kmeans=data.preprocesing(OrdinalEncoder(),MinMaxScaler())
df['clusters']=data.work(KMeans(n_clusters=4),kmeans,4,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])

st.title('If you want , you can experiment yourself')
st.write("Let's choose different clustering models")
models = ['KMeans', 'Aglomeriv']
selected_models = st.multiselect('Choose a model for clustering', models)

if 'KMeans' in selected_models:
  pre=['StandardScaler','RobustScaler','MinMaxScaler']
  preproces=st.multiselect('Select a model for preprocessing features', pre)

  if 'StandardScaler' in preproces:
    data=claster(df)
    kmeans=data.preprocesing(OrdinalEncoder(),StandardScaler())
    num_of_clusters=st.number_input('select the number of clusters', min_value=2, max_value=12)
    data.work(KMeans(n_clusters=num_of_clusters),kmeans,num_of_clusters,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])  
    if st.button('Use it ?'): 
      df['clusters']=data.work(KMeans(n_clusters=num_of_clusters),kmeans,num_of_clusters,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])

  if 'RobustScaler' in preproces:
    data=claster(df)
    kmeans=data.preprocesing(OrdinalEncoder(),RobustScaler())
    num_of_clusters=st.number_input('select the number of clusters', min_value=2, max_value=12)
    data.work(KMeans(n_clusters=num_of_clusters),kmeans,num_of_clusters,[silhouette_score,davies_bouldin_score,calinski_harabasz_score]) 
    if st.button('Use it ?'): 
      df['clusters']=data.work(KMeans(n_clusters=num_of_clusters),kmeans,num_of_clusters,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])

  if 'MinMaxScaler' in preproces:
    data=claster(df)
    kmeans=data.preprocesing(OrdinalEncoder(),MinMaxScaler())
    num_of_clusters=st.number_input('select the number of clusters', min_value=2, max_value=12)
    data.work(KMeans(n_clusters=num_of_clusters),kmeans,num_of_clusters,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])
    if st.button('Use it ?'): 
      df['clusters']=data.work(KMeans(n_clusters=num_of_clusters),kmeans,num_of_clusters,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])

if 'Aglomeriv' in selected_models:
  pre=['StandardScaler','RobustScaler','MinMaxScaler']
  preproces=st.multiselect('Select a model for preprocessing features', pre)

  if 'StandardScaler' in preproces:
    data=claster(df)
    aglo=data.preprocesing(OrdinalEncoder(),StandardScaler())
    num_of_clusters=st.number_input('select the number of clusters', min_value=2, max_value=12)
    st.text('Выебри метод :"ward","complete","single","average"')
    metod=st.text_input(' Method :')
    data.work(AgglomerativeClustering(n_clusters=num_of_clusters, linkage=metod,compute_full_tree=True),aglo,2,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])
    if st.button('Use it ?'):
      df['clusters']=data.work(AgglomerativeClustering(n_clusters=num_of_clusters, linkage=metod,compute_full_tree=True),aglo,2,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])
      
  if 'RobustScaler' in preproces:
    data=claster(df)
    aglo=data.preprocesing(OrdinalEncoder(),RobustScaler())
    num_of_clusters=st.number_input('select the number of clusters', min_value=2, max_value=12)
    st.text('Выебри метод :"ward","complete","single","average"')
    metod=st.text_input(' Method :')
    data.work(AgglomerativeClustering(n_clusters=num_of_clusters, linkage=metod,compute_full_tree=True),aglo,2,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])
    if st.button('Use it ?'):
      df['clusters']=data.work(AgglomerativeClustering(n_clusters=num_of_clusters, linkage=metod,compute_full_tree=True),aglo,2,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])

  if 'MinMaxScaler' in preproces:
    data=claster(df)
    aglo=data.preprocesing(OrdinalEncoder(),MinMaxScaler())
    num_of_clusters=st.number_input('select the number of clusters', min_value=2, max_value=12)
    st.text('Выебри метод :ward,complete,single,average')
    metod=st.text_input(' Method :')
    data.work(AgglomerativeClustering(n_clusters=num_of_clusters, linkage=metod,compute_full_tree=True),aglo,2,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])
    if st.button('Use it?'):
      df['clusters']=data.work(AgglomerativeClustering(n_clusters=num_of_clusters, linkage=metod,compute_full_tree=True),aglo,2,[silhouette_score,davies_bouldin_score,calinski_harabasz_score])


new_df=df
new_df['clusters']=new_df['clusters'].astype('category')





bar_columns=['Education','Marital_Status','Year_Birth']

st.write('The ratio of the resulting clusters')
st.table(new_df['clusters'].value_counts(normalize=True))

def rename(data):
  if data['clusters'] == 0:
    return 'Group_1'
  elif data['clusters']==1:
    return 'Group_2'
  elif data['clusters'] ==3:
    return 'Group_3'
  else:
    return 'Group_4'

new_df['clusters']=new_df.apply(rename,axis=1)

st.title('Task 2: To analyze the resulting groups.')
st.write("Let's divide our columns into several groups for analysis :")
st.text('A first group of nominative features that can provide general information about customers')
st.text('A second group of attributes that display purchases of groups of certain products')
st.text('A third group that will display the number of purchases where they are perfect and activity')

first_category=['Year_Birth', 'Education', 'Marital_Status', 'Income','Basket','Childs']
second_category=['MntWines','MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts','MntGoldProds']
third_category=['NumWebVisitsMonth','NumWebPurchases', 'NumCatalogPurchases','NumStorePurchases',
                'procent_offline','count_of_purchases']




st.title('First group')

for i in first_category:
  if i not in ['Income','Basket','Year_Birth']:
    plt.figure(figsize=(8,8))
    data=new_df.pivot_table(index=i,columns='clusters',values='MntSweetProducts',aggfunc='count')
    pivot_df = data.reset_index()
    melted_df = pivot_df.melt(id_vars=i, value_name='count', var_name='clusters')
    sns.barplot(x=i, y='count', hue='clusters', data=melted_df, palette=my_colors)
    st.pyplot()
    plt.show()
  elif i=='Basket':
    plt.figure(figsize=(8,8))
    sns.kdeplot(data=new_df,x=new_df.loc[new_df[i]<10000][i],palette=my_colors,hue='clusters')
    plt.axvline(new_df['Basket'].median(), linestyle='--', color='red')
    plt.text(new_df['Basket'].median(),0,'Медиана корзины')
    plt.show()
    plt.figure(figsize=(8,8))
    sns.jointplot(x=new_df[i],y=new_df['Income'],hue=new_df['clusters'],kind='kde',palette=my_colors,alpha=0.6)
    st.pyplot()
    plt.show()
  elif i=='Year_Birth':
    plt.figure(figsize=(8,8))
    sns.kdeplot(data=new_df,x=new_df[i],palette=my_colors,hue='clusters',common_norm=False)
    st.pyplot()
  else:
    plt.figure(figsize=(8,8))
    sns.kdeplot(data=new_df,x=new_df[i],palette=my_colors,hue='clusters')
    st.pyplot()
    plt.show()

st.write("""Conclusion about first part""")
st.subheader("Age:")
st.write("- Group 1 is the oldest of their age before 1961")
st.write("- Group 2 comes after from 1958 to 1972")
st.write("- Group 3 goes from 1978 to 1982")
st.write("- The last 4 groups are from 1988 to 2000")
st.write("")
  # Education
st.subheader("Education:")
st.write("- The most educated is Group 2, followed by Group 3")
st.write("- The less educated group is the 4th group")
st.write("")

# Marital status
st.subheader("Marital status:")
st.write("- Most of all married in the first group and in 3")
st.write("- The least divorced in the 4th group may well be related to age")
st.write("")

# Profit
st.subheader("The profit that the groups bring:")
st.write("- Group 1 brings the least about 39 thousand")
st.write("- Group 2 on average brings from 45 thousand to 75 thousand")
st.write("- Group 3 on average brings 60 thousand and this is the best result since most of the people are focused here")
st.write("- As for Group 4, there is a clear division by part of this group on average bring about 30 thousand, and the other from 75 thousand to 80 thousand. This is considered the highest indicator, but the number of people here is not as large as the third group.")
st.write("")

# Average basket
st.subheader("The average basket:")
st.write("- The first group, we see that a large accumulation of about 3000 thousand and interestingly, with a small average basket, they generally bring about 70 thousand")
st.write("- Group 2 we see two points of accumulation, the first average basket is slightly above 6000 thousand, but they bring about 30 thousand total income, but at the same time the second point which has an average basket of 4500, but they bring much more from 55 to 75 thousand")
st.write("- As with the incomes of Group 3, the average basket is focused on 3 thousand with a total income of about 60 thousand")
st.write("- The last 4 group has as many as three points of accumulation, 1 point and 2 point is the average basket of 3 thousand and 6 thousand with an average income of no more than 30 thousand and 3 point is also 3 thousand of the average basket, but at the same time they bring about 70 thousand.")
st.write("")

# Children
st.subheader("Children:")
st.write("- We see that the 4 groups most often do not have children, and they have no more than 2 as much as possible")
st.write("- People from the second group have one child more often than others")
st.write("- People from groups 1 and 3 are the most large")

st.title('Second group')
for i in second_category:
  
  sns.jointplot(x=new_df[i],y=new_df['Income'],hue=new_df['clusters'],kind='kde',palette=my_colors,alpha=0.6)
  st.pyplot()

st.write('')
# Alcohol sales
st.subheader("Alcohol sales:")
st.write("- Group 1 is ready to spend more than all other groups")
st.write("- Group 2 goes after it")
st.write("- The least buys alcohol 4 group")
st.write("- Then comes the third group")
st.write("")
st.write("We also see that, in general, the trend is that the more alcohol is bought, the more profit is made, but the main group of people is from 0 to 250 units.")
st.write("")

# Fruit sales
st.subheader("Fruit sales:")
st.write("- People from the 4th group buy fruit the most and at the same time bring the most profit")
st.write("")

# Meat sales
st.subheader("Meat sales:")
st.write("- The main sales for all groups are approximately in the same range from 0 to 250 units")
st.write("- The first group is actively allocated here, which has purchases in large quantities, there are purchases of 600 and 800 units")
st.write("- Rarely the 2nd group buys meat more than normal")
st.write("")

# Fish sales
st.subheader("Fish sales:")
st.write("- The trend keeps from 0 to 50 units in almost all groups")
st.write("- 4 groups have people who buy more than 100 units of fish")
st.write("")

# Sweet goods sales
st.subheader("Sweet goods sales:")
st.write("- Everyone buys in about the same range")
st.write("- 2 and 1 groups buy in larger quantities than the rest")
st.write("")

# Promotional products sales
st.subheader("Promotional products sales:")
st.write("- Promotional products are more popular with the 2nd group with a profit of up to 60 thousand, and the first group of people")
st.write("- The 4 groups that buy up to 50 units bring the company the most money")

st.title('Third group')
for i in third_category:
  if i not in ['procent_offline','count_of_purchases']:
    plt.figure(figsize=(10,7))
    sns.barplot(x=new_df[i],y=new_df['Income'],hue=new_df['clusters'], palette=my_colors)
    st.pyplot()
  else:
    plt.figure(figsize=(10,7))
    sns.jointplot(x=new_df[i],y=new_df['Income'],hue=new_df['clusters'],kind='kde',palette=my_colors,alpha=0.6)
    st.pyplot()

st.subheader("Site visit:")
st.write("- We see that the 1st and 4th group visited the site at least once")
st.write("- People of the 2nd and 3rd groups can do without visiting the site")
st.write("- The 3rd group most often visits the site 10 times")
st.write("- Group 4 makes the most purchases on the site")
st.write("- People who make purchases on the site bring the company no more than 65 thousand")
st.write("")

# Catalog usage
st.subheader("Catalog usage:")
st.write("- The catalog is more popular with the first group")
st.write("- People from the 4th group did not make more than 10 purchases on the catalog")
st.write("- More affluent people use the catalog")
st.write("- As from 3 to 10 purchases, the profit grows from 50 thousand to 80")
st.write("")

# Purchases in a physical store
st.subheader("Purchases in a physical store:")
st.write("- Group 4 makes purchases more often")
st.write("- But at the same time, they do not bring a lot of money")
st.write("- A small part of people from Group 2 did not make purchases in a physical store")
st.write("")

# Offline buyers
st.subheader("Offline buyers:")
st.write("- Some people from the 2nd and 3rd groups have never used online platforms when shopping")
st.write("- But at the same time, these categories do not bring much money")
st.write("- In the 1st group, we see that they make purchases less often in a physical store, but at the same time the profit they make is about 60 thousand")
st.write("- Group 4 can also be divided into two groups, those who use online more often and they spend more money, and those who are less likely to bring about 30 thousand on average")


color_map={'Group_1': 'red', 'Group_2': 'blue', 'Group_3': 'green','Group_4':'yellow'}

st.title('Conduct your own research')

if st.checkbox('scatter'):
      columns_1=new_df.columns
      selected_column_1 = st.selectbox('Select a column for the axis X ', columns_1)
      columns_2=new_df.columns
      selected_column_2 = st.selectbox('Select a column for the axis Y', columns_2)
      fig=px.scatter(new_df, y=selected_column_2,x=selected_column_1, marginal_x='histogram', marginal_y='histogram',color='clusters',color_discrete_map=color_map)
      
      st.plotly_chart(fig)

if st.checkbox('histogram'):
      columns_x=new_df.columns
      selected_column_x = st.selectbox('Select a column for the axis X ', columns_x)
      fige = px.histogram(new_df, x=selected_column_x, nbins=100, opacity=0.7,color='clusters',color_discrete_map=color_map)
      fige.update_layout(xaxis_rangeslider_visible=True)
      st.plotly_chart(fige) 

st.title('Task 3: Draw conclusions and make suggestions.') 

st.subheader("Group 1:")
st.write("- Older demographic")
st.write("- Mostly married")
st.write("- Low profit generation")
st.write("- High average basket value")
st.write("- Prefers purchasing alcohol")
st.write("- Less likely to use online platforms")
st.write("")
st.write("Recomended:")
st.write("- Implement targeted promotions for alcohol, such as discounts, combo offers, or loyalty programs.")
st.write("- Provide in-store events or experiences that appeal to their age group and preferences.")
st.write("- Use traditional marketing channels like print ads, radio, or direct mail to reach them.")
st.write("- Offer a senior discount program or age-specific loyalty program that rewards them for repeat purchases.")
st.write("- Keep them informed of new alcohol products or limited-time offers through direct mail or print newsletters.")
st.write("- Provide exceptional in-store customer service and assistance, catering to their preferences and needs.")
st.write("")

# Group 2
st.subheader("Group 2:")
st.write("- Middle-aged demographic")
st.write("- Highly educated")
st.write("- Moderate profit generation")
st.write("- Varying average basket value")
st.write("- Prefers purchasing alcohol and promotional products")
st.write("- More likely to use online platforms")
st.write("")
st.write("Recomended:")
st.write("- Offer special deals on alcohol and promotional products, targeting their preferences.")
st.write("- Leverage their education by sharing informative content about products and their benefits.")
st.write("- Use a mix of traditional and digital marketing channels to reach this audience.")
st.write("- Share informative content related to their product preferences.")
st.write("- Use a mix of traditional and digital communication channels to maintain engagement.")
st.write("- Offer exclusive deals and promotions tailored to their preferences and education level.")
st.write("")

# Group 3
st.subheader("Group 3:")
st.write("- Middle-aged demographic")
st.write("- Married")
st.write("- High profit generation")
st.write("- Low average basket value")
st.write("- Similar preferences to Group 1")
st.write("")
st.write("Recomended:")
st.write("- Implement family-oriented promotions, considering their marital status and higher spending power.")
st.write("- Offer bundle deals or discounts to encourage increased spending per transaction.")
st.write("- Use both digital and traditional marketing channels to engage with them effectively.")
st.write("- Send personalized offers and promotions based on their purchase history and preferences.")
st.write("- Use family-oriented events or in-store experiences to keep them engaged and interested in your brand.")
st.write("- Leverage both digital and traditional marketing channels to maintain contact and encourage repeat purchases.")
st.write("")

# Group 4
st.subheader("Group 4:")
st.write("- Younger demographic")
st.write("- Less educated")
st.write("- High profit generation with potential")
st.write("- Prefers purchasing fruits")
st.write("- Highly engaged with online store and catalog")
st.write("")
st.write("Recomended:")
st.write("- Encourage fruit purchases by offering discounts, bundles, or seasonal promotions.")
st.write("- Invest in improving the online shopping experience and promote exclusive online deals.")
st.write("- Utilize digital marketing channels, such as social media, email marketing, and influencer partnerships.")
st.write("- Implement a digital loyalty program that rewards them for repeat online purchases and catalog usage.")
st.write("- Use email marketing and social media to share exclusive online deals, seasonal fruit promotions, and relevant content.")
st.write("- Offer personalized product recommendations and a seamless online shopping experience to keep them engaged and loyal to your brand.")
