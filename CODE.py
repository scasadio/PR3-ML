#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np


# In[1]:


The project is based on the analysis of the spread of the COVID-19 in the countees of the United Stetes.
It is a county-level analysis based on demographic data (race, sex, migration behaviour, health capacity...), 
observed confirmed cases of infected and death people, 
mobiity features (changes in the mobilty of the population with respect to the baseline scores), 
and weather data (humidity and temperature expressed monthly).


# In[102]:


demographic = pd.read_csv("C:\\Users\\Utente\\OneDrive\\Desktop\\pr3\\dati veri\\demografia\\demographic.csv")
cases = pd.read_csv("C:\\Users\\Utente\\OneDrive\\Desktop\\pr3\\dati veri\\demografia\\infected_death_cases.csv").drop(["AREA_NAME", "STATE"], axis=1)
data_mobility=pd.read_csv("C:\\Users\\Utente\\OneDrive\\Desktop\\pr3\\dati veri\\demografia\\2020_US_Region_Mobility_Report.csv")
data_weather=pd.read_csv("C:\\Users\\Utente\\OneDrive\\Desktop\\pr3\\dati veri\\clima\\unified_climate.csv")


# In[15]:


These line of code manipulate the DataFrame 'new_mobility' to extract and group data by month and census FIPS code. 
First, the 'date' column is converted to a datatime data type.
Next, a new column called 'month'is created by extracting the month component from the 'date' column using "dt.month" attribute.
Then, both the 'data' column and the rows corresponding to months greater than September are dropped. 
Finally, the "data1" DataFrame is grouped by census FIPS code and month using the ".groupby()" method, and the mean is calculated for the
columns related to mobility behavior during the COVID-19 pandemic.


# In[104]:


new_mobility= data_mobility.drop(['country_region_code','country_region','sub_region_1','sub_region_2',
                                  'metro_area','iso_3166_2_code', 'place_id'],axis=1)
new_mobility['date']= pd.to_datetime(new_mobility['date'])
new_mobility['mese']= new_mobility['date'].dt.month
mobility2= new_mobility.drop(['date'],axis=1)
data1= mobility2.drop(new_mobility[mobility2['mese'] > 9].index)


# In[17]:


We grouped data using the groupby method based on 2 variables,(census fips code and date)
then we compute the mean of the values for the other columns.
Using 'reset_index' function we reset the index and replaced it with new indices.


# In[105]:


data_group = data1.groupby(['census_fips_code','mese']).mean(['retail_and_recreation_percent_change_from_baseline',
                        'grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline',
                       'transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline',
                      'residential_percent_change_from_baseline']).reset_index()


# In[17]:


These lines of code are related to feature scaling and data cleaning.
After dropping the columns 'census_fips_code' and 'month', we create a new dataset containing the scaled variables.
Due to the high number of missing values, we used the dropna() method.
The parameter 'inplace=True' indicates that the operation should be performed on the original DataFrame.
When we set axis=1 and how='any', we remove any columns (axis=1) that contain at least one missing value (how='any'). Similarly, 
when we set axis=0 and how='any', we remove any rows (axis=0) that contain at least one missing value (how='any'). 


# In[106]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[107]:


mobility_sd = pd.DataFrame(scaler.fit_transform(data_group.drop(['census_fips_code','mese'],axis=1)), columns=data1.drop(['census_fips_code','mese'],axis=1).columns)
mobility_sd.dropna(axis=0, how='any', inplace=True)  # Elimina le righe con valori mancanti
mobility_sd.dropna(axis=1, how='any', inplace=True)  # Elimina le colonne con valori mancanti


# In[20]:


We performe the PCA on 'mobility_sd' DataFrame in order to reduce data by projecting them in a lower dimensional space.
We calculate the percentage of variance explained by each principal component, which can be useful in 
determining the appropriate number of principal components to consider.


# In[108]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[109]:


pca1 = PCA()
pca1.fit(mobility_sd)
cum_var= np.sum(pca1.explained_variance_ratio_[:2])* 100
print(f'Cumulate percentage of explained variance of the model, 2 PC. {cum_var:.2f}%')


# In[21]:


The two principal components are sufficient to explain a high percentage of variance, (86.12%),
so it may be reasonable to use them to represent the original dataset in a lower dimensional space. 


# In[110]:


pca = PCA(n_components=2)
mobpca = pd.DataFrame(data=pca.fit_transform(mobility_sd), columns=['PC1', 'PC2'])


# In[25]:


We choose 2 as numeber of principal componets 

The 'fit_transform' function performs two steps:
first, it trains the PCA model on the DataFrame data
then, it transforms the data into a new coordinate system in which the columns are called 'PC1' and 'PC2'.


# In[163]:


cum_var1= np.cumsum(pca1.explained_variance_ratio_)
plt.plot(cum_var1),plt.xlabel('# principal components-1'), plt.ylabel('Cumulative sum of variance'),plt.title('Cumulative Variance plot'), plt.xlim(1,5)
plt.show()


# In[27]:


Here, we can visualize the cumulative variance explained by each principal component. 
The x-axis shows the number of principal components, while the y-axis shows the cumulative sum of explained variance.

We do the same procedure with the climate dataset.


# In[111]:


weather= data_weather.drop(['FIPS'],axis=1)
weather_sd = pd.DataFrame(scaler.fit_transform(weather), columns=weather.columns)


# In[112]:


pca_w1 = PCA()
pca_w1.fit(weather_sd)
cum_var2= np.sum(pca_w1.explained_variance_ratio_[:3])* 100
print(f'Cumulate percentage of explained variance of the model, 3 PC. {cum_var2:.2f}%')


# In[113]:


pca_w = PCA(n_components=3)
weatherpca = pd.DataFrame(data=pca_w.fit_transform(weather_sd), columns=['PC1 weater', 'PC2 weater','PC3 weater'])


# In[162]:


cum_var3= np.cumsum(pca_w1.explained_variance_ratio_)
plt.plot(cum_var3),plt.xlabel('# principal components-1'), plt.ylabel('Cumulative sum of variance'),plt.title('Cumulative Variance plot'), plt.xlim(1,3)
plt.show()


# In[114]:


pca= pd.concat([mobpca,weatherpca],axis=1)


# In[115]:


merged = pd.merge(demographic, cases, on='FIPS')
merged1=merged.drop(['FIPS','State','Area_Name','Rural-urban_Continuum Code_2013','Urban_Influence_Code_2013',
                    'Economic_typology_2015', 'Unnamed: 0'], axis=1)


# In[119]:


merge1 = scaler.fit_transform(merged1)
merge1sd = pd.DataFrame(merge1, columns=merged1.columns)
mergedsd = pd.concat([merged[['FIPS','State','Area_Name','Economic_typology_2015']],merge1sd], axis=1)


# In[120]:


pca_final =pd.concat([demographic[['FIPS']],pca],axis=1)
dataset_final = pd.merge(mergedsd, pca_final, on='FIPS')


# In[121]:


y= dataset_final['infected']
X =dataset_final.drop(['State','Area_Name','infected'],axis=1).fillna(dataset_final.drop(['State','Area_Name','infected'],axis=1).mean())
X.fillna(X.mean(),inplace= True)


# In[122]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[93]:


#RANDOM FOREST
It is an estimator for supervised learning that fits a number of decision trees on various subsamples of 
the dataset that uses average to improve the accuracy of estimates.

The code below trains a random forest regression model using the RandomForestRegressor function. The hyperparameters of the model have a specif meaning, such as:
"n_estimators" indicates the number of trees in the forest.
"max_features="auto"" indicates the maximum number of features to consider when splitting a node, here it is set to the square root of the total number of features.
"random_state=42" is the random seed, it ensure reproducibility.
"max_depth=3" the maximum depth of each tree.
 
At the end of the code, we evaluate the model performance using 'score' method, which returns the coefficient of determination R^2 of the prediction.


# In[127]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree


# In[128]:


rfr = RandomForestRegressor(n_estimators=100, max_depth=3, max_features="auto", random_state=42)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
print("Accuracy:", rfr.score(X_test, y_test))


# In[62]:


The accuracy is the degree of similarity we obtain between the real value and the value resulting from the sample data.
It is a value bounded between 0 and 1, where if 1 we obtain the estimated value equal to the true one.


# In[129]:


plt.figure(figsize=(20, 20))
plot_tree(rfr.estimators_[0], filled=True)
plt.show()


# In[44]:


On the far right we have one sample obtained by considering X[165]<=4.09, X[189]<=2.249, X[121]<=1.746.
The sample units have mean =14.178 and mean squared error =0.0 . This latter one tells us that the prediction for the sample
are equals to the true values.


# In[ ]:





# In[45]:


#GRADIENT BOOSTING
It constructs a sequence of simple decision trees in which each tree is built based on the reults of the previous tree predictive error.
It relies on the intuition that the best possible next model, when combined with previous models, 
minimizes the overall prediction error. 
We aim to fit the gradient boosting regressor on the training data in order to end up with the best parittion 
of data based on the Friedman means squared error.
This one is the mse with improvement score by Friedman, that allows to produce better approximations.


# In[130]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# In[134]:


gbr = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=100, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
print("Accuracy:", gbr.score(X_test, y_test))
print(f'Mean squared error: {mean_squared_error(y_test, y_pred):.2f}%')


# In[160]:


plt.figure(figsize=(20, 20))
plot_tree(gbr.estimators_[0][0], filled=True)
plt.show()


# In[73]:


On the far right we obtain 4 samples that have mean =8.191, that are found by considering the splits X[29]>31.322,
X[185]<=0.863, X[144]<=1.1413.
The Friedman mse is not sufficliently low to say that we end up with a good prediction for these samples.


# In[50]:


#KMEANS
It is an unsipervised learning method, so a clustering algorithm that aims to find the best number
of group in order to minimize the squarred error between the mean of a cluster and the observation within 
that cluster.
We produce a kmeans analysis on a range of possibles values for k that goes from 2 to 14, and we select the 
best one according to the silhouette index.


# In[141]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[148]:


sse = []
possible_n_clusters = range(2, 15)
for k in possible_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)


# In[158]:


plt.plot(possible_n_clusters, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')


# In[145]:


silhouette_scores = []
for n_clusters in possible_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))


# In[157]:


plt.plot(possible_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')


# In[157]:


This code calculates the Silhouette score for a K-means clustering algorithm with 7 clusters. 
The "n_clusters" parameter specifies the number of clusters to be created.
The "labels_" attribute of the kmeans object returns the cluster labels for each data point.


# In[150]:


kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(X)
print('Silhouette score:', silhouette_score(X, kmeans.labels_))


# In[55]:


The silhouette index is a measure of the quality of a partition of dataset and it is computed by the
difference between the average distance between a point and points in the same cluster, and the average distance
between our point and points in near different clusters.
It is bounded between -1 and 1, and the higher the better the classification of points.


# In[55]:


The Silhouette score is about 0.67, this is a relatively high value that indicating that the clusters are well-separated and compact, with a clear boundary between them.


# In[151]:


labels= kmeans.fit_predict(X)
print(labels)


# In[172]:


X_new= X.iloc[:, :].values
plt.scatter(X_new[:, 0], X_new[:,1], c=labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.5)


# In[172]:


In the plot above, the color of each point is determined by the "labels".
The "s" parameter controls the size of the plotted points, and the "cmap" parameter specifies the color map.


# In[58]:


#HIERARCHICAL- WARD METHOD
The hierarchical methods create sequences of partitions as homogeneous as possibile within and as hererogeneous as possible beteween. 
Here we use an agglomerative clustering algorithm that merges clusters according to the ward method linkage.
Ward aims to group clusters that minimises the sum of squared error.
We apply the method and produce a dendrogram.


# In[175]:


import scipy.cluster.hierarchy as hy
from scipy.cluster.hierarchy import dendrogram


# In[176]:


Z = linkage(X, 'ward')
fig = plt.figure(figsize=(15, 15))
dn = dendrogram(Z)
plt.title('Ward Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


# In[177]:


From this dendrogram we can note that the best paritition seems to be k=9 clusters.

