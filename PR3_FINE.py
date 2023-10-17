#!/usr/bin/env python
# coding: utf-8

# In[235]:


import numpy as np
import pandas as pd


# The project is based on the analysis of the spread of the COVID-19 in the countees of the United Stetes.
# It is a county-level analysis based on demographic data (race, sex, migration behaviour, health capacity...), 
# observed confirmed cases of infected and death people, mobiity features (changes in the mobilty of the population with respect to the baseline scores), and weather data (humidity and temperature expressed monthly).

# In[237]:


demographic = pd.read_csv("C:\\Users\\carol\\Desktop\\PR3_CarolaSara\\demographic.csv").drop(['Unnamed: 0'],axis=1)
cases = pd.read_csv("C:\\Users\\carol\\Desktop\\PR3_CarolaSara\\infected_death_cases.csv").drop(["AREA_NAME", "STATE"], axis=1)

data_mobility= pd.read_csv("C:\\Users\\carol\\Desktop\\PR3_CarolaSara\\2020_US_Region_Mobility_Report.csv")
data_weather= pd.read_csv("C:\\Users\\carol\\Desktop\\PR3_CarolaSara\\unified_climate.csv") 


# In[ ]:


new_mobility= data_mobility.drop(['country_region_code','country_region','sub_region_1','sub_region_2',
                                  'metro_area','iso_3166_2_code', 'place_id'],axis=1)


# These line of code manipulate the DataFrame 'new_mobility' to extract and group data by month and census FIPS code. 
# First, the 'date' column is converted to a datatime data type.
# Next, a new column called 'month'is created by extracting the month component from the 'date' column using "dt.month" attribute.
# Then, both the 'data' column and the rows corresponding to months greater than September are dropped. 
# Finally, the "data1" DataFrame is grouped by census FIPS code and month using the ".groupby()" method, and the mean is calculated for the columns related to mobility behavior during the COVID-19 pandemic.

# In[238]:


new_mobility['date']= pd.to_datetime(new_mobility['month'])
new_mobility['month']= new_mobility['month'].dt.month
mobility2= new_mobility.drop(['month'],axis=1)
data1= mobility2.drop(new_mobility[mobility2['month'] > 9].index)

data_group = data1.groupby(['census_fips_code','month']).mean(['retail_and_recreation_percent_change_from_baseline',
                        'grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline',
                       'transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline',
                      'residential_percent_change_from_baseline']).reset_index()


# These lines of code are related to feature scaling and data cleaning.
# After dropping the columns 'census_fips_code' and 'month', we create a new dataset containing the scaled variables.
# Due to the high number of missing values, we used the dropna() method.
# The parameter 'inplace=True' indicates that the operation should be performed on the original DataFrame.
# When we set axis=1 and how='any', we remove any columns (axis=1) that contain at least one missing value (how='any'). Similarly, when we set axis=0 and how='any', we remove any rows (axis=0) that contain at least one missing value (how='any'). 
# 

# In[239]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
mobility_sd = pd.DataFrame(scaler.fit_transform(data_group.drop(['census_fips_code','mese'],axis=1)), columns=data1.drop(['census_fips_code','mese'],axis=1).columns)
mobility_sd.dropna(axis=0, how='any', inplace=True)
mobility_sd.dropna(axis=1, how='any', inplace=True)


# We performe the PCA on 'mobility_sd' DataFrame and calculates the percentage of variance, which can be useful in determining the appropriate number of principal components to include in the model.

# In[240]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#PCA 
pca1 = PCA()
pca1.fit(mobility_sd)
cum_var= np.sum(pca1.explained_variance_ratio_[:2])* 100
print(f'Cumulate percentage of explained variance of the model, 2 PC. {cum_var:.2f}%')


# The two principal components explain a high percentage of variance, (86.12%), so it may be reasonable to use them to represent the original dataset in a lower dimensional space. 

# In[241]:


#PCA mobility 
pca = PCA(n_components=2)
mobpca = pd.DataFrame(data=pca.fit_transform(mobility_sd), columns=['PC1', 'PC2'])


# We choose 2 as numeber of principal componets 

# The 'fit_transform' function performs two steps:
# first, it trains the PCA model on the DataFrame data
# then, it transforms the data into a new coordinate system in which the columns are called 'PC1' and 'PC2'.

# In[242]:


cum_var1= np.cumsum(pca1.explained_variance_ratio_)
plt.plot(cum_var1),plt.xlabel('# principal components-1'), plt.ylabel('Cumulative sum of variance'),plt.title('Cumulative Variance plot'), plt.xlim(1,5)


# Here, we can visualize the cumulative variance explained by each principal component. The x-axis shows the number of principal components, while the y-axis shows the cumulative sum of explained variance.

# We do the sema procedure with the climate dataset

# In[272]:


weather= data_weather.drop(['FIPS'],axis=1)
weather_sd = pd.DataFrame(scaler.fit_transform(weather), columns=weather.columns)


# In[273]:


#PCA weater
pca_w1 = PCA()
pca_w1.fit(weather_sd)

#Cumulative percentage of explained variance of the model 
cum_var2= np.sum(pca_w1.explained_variance_ratio_[:3])* 100
print(f'Cumulate percentage of explained variance of the model, 3 PC. {cum_var2:.2f}%')


# The tree principal components explain a high percentage of variance, (84.80%), so it may be reasonable to use them to represent the original dataset a lower dimensional space. 

# In[274]:


pca_w = PCA(n_components=3)
weatherpca = pd.DataFrame(data=pca_w.fit_transform(weather_sd), columns=['PC1 weather', 'PC2 weather','PC3 weather'])


# The 'fit_transform' function performs two steps:
# first, it trains the PCA model on the DataFrame data
# then, it transforms the data into a new coordinate system in which the columns are called 'PC1 weather', 'PC2 weather' and 'PC3 weather'.

# In[259]:


#Cumulative variance plot
cum_var3= np.cumsum(pca_w1.explained_variance_ratio_)
plt.plot(cum_var3),plt.xlabel('# principal components-1'), plt.ylabel('Cumulative sum of variance'),plt.title('Cumulative Variance plot'), plt.xlim(1,3)


# Here, we can visualize the cumulative variance explained by each principal component in weather setting. The x-axis shows the number of principal components, while the y-axis shows the cumulative sum of explained variance.
# We choose 3 as number of principal component.

# In[260]:


pca= pd.concat([mobpca,weatherpca],axis=1)


# In this line of code we combine the demographic information with the COVID-19 cases information for the same geographical areas identified by the FIPS code,(on=FIPS).

# In[305]:


merged = pd.merge(demographic, cases, on='FIPS')


# In[275]:


merged1=merged.drop(['FIPS','State','Area_Name','Rural-urban_Continuum Code_2013','Urban_Influence_Code_2013',
                     'Economic_typology_2015'], axis=1)


# In[276]:


merge1 = scaler.fit_transform(merged1)
merge1sd = pd.DataFrame(merge1, columns=merged1.columns)
mergedsd = pd.concat([merged[['FIPS','State','Area_Name','Economic_typology_2015']],merge1sd], axis=1)


# In[278]:


pca_final =pd.concat([demographic[['FIPS']], pd.concat([pca],axis=1)], axis=1)
dataset_final = pd.merge(mergedsd, pca_final, on='FIPS')


# This code creates the feature matrix X and target variable y for a machine learning model.
# In addition, we replace any missing values in X with the mean of the non-missing values in each column.

# In[279]:


y= dataset_final['infected']
X =dataset_final.drop(['State','Area_Name','infected'],axis=1).fillna(dataset_final.drop(['State','Area_Name','infected'],axis=1).mean())
X.fillna(X.mean(),inplace= True)


# In these line of code we perfome a split of the data into train and test sets using "train_test_split" function.
# The parameter 'random_state' is used to set the random seed for reproducibility, it ensuring that the same split is obtained each time code is run.

# In[280]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[281]:


#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree


# It is an estimator for supervised learning that fits a number of decision trees on various subsamples of 
# the dataset that uses average to improve the accuracy of estimates

# The code below trains a random forest regression model using the RandomForestRegressor function. The hyperparameters of the model have a specif meaning, such as:
# "n_estimators" indicates the number of trees in the forest.
# "max_features="auto"" indicates the maximum number of features to consider when splitting a node, here it is set to the square root of the total number of features.
# "random_state=42" is the random seed, it ensure reproducibility.
# "max_depth=3" the maximum depth of each tree.
# 
# At the end of the code, we evaluate the model performance using 'score' method, which returns the coefficient of determination R^2 of the prediction. The R^2 score measures the proportion of variance in the target variable that is explained by the model.

# In[282]:


rfr = RandomForestRegressor(n_estimators=100, max_features="auto", random_state=42, max_depth= 3)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
accuracy = rfr.score(X_test, y_test)
print("Accuracy:", accuracy)


# As we can see, from the code above we obtain a negative R^2 score. 
# This indicates that the model performs worse than a model that simply predicts the mean value of the target variable.
# Among the possible causes of this result, there could be:
# 
# Underfitting: The model may be too simple to capture the underlying patterns in the data, resulting in underfitting. Underfitting occurs when the model is not able to capture the complexity of the data.
# 
# Missing values: Replacing missing values can significantly affect the performance of the model, as missing values can alter the distributions of variables and introduce bias into the data.

# In[268]:


plt.figure(figsize=(30, 30))
plot_tree(rfr.estimators_[0], filled=True)
plt.show()


# On the far right we have one sample obtained by considering X[165]<=4.09, X[189]<=2.249, X[121]<=1.746.
# The sample units have mean =14.178 and mean squared error =0.0 . This latter one tells us that the prediction fo the sample are equale to the true values.

# In[283]:


#GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# It constructs a sequence of simple decision trees in which each tree is built based on the reults of the
# previous tree predictive error.
# It relies on the intuition that the best possible next model, when combined with previous models, 
# minimizes the overall prediction error. 
# We aim to fit the gradient boosting regressor on the training data in order to end up with the best parittion 
# of data based on the Friedman means squared error.
# This one is the mse with improvement score by Friedman, that allows to produce better approximations.

# In[295]:


gbr = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=100, random_state=42, max_depth=3) 
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
accuracy = gbr.score(X_test, y_test)
print("Accuracy:", accuracy)
print(f'Mean squared error: {mse:.2f}%')


# 

# In[285]:


plt.figure(figsize=(20, 20))
plot_tree(gbr.estimators_[0][0], filled=True)
plt.show()


# On the far right we obtain 4 samples that have mean =8.191, that are found by considerig the splits X[29]>31.322,
# X[185]<=0.863, X[144]<=1.1413.
# The Friedman mse is no sufficliently low to say that we end up with a good prediction for these samples.

# In[296]:


#KMEANS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# It is an unsipervised learning method, so a clustering algorithm that aims to find the best number
# of group in order to minimize the squarred error between the mean of a cluster and the observation within 
# that cluster. We produce a kmeans analysis on a range of possibles values for k that goes from 2 to 14, and we select the 
# best one according to the silhoette index.

# In[297]:


sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)


plt.plot(range(2, 15), sse, marker='o')
plt.xlabel('Numero di cluster')
plt.ylabel('SSE')
plt.show()


# A line plot is generated with K values on the x-axis and SSE values on the y-axis.
# This plot helps to identify the elbow point, which is the point where the SSE starts to decrease at a slower rate. This elbow point helps in determining the optimal number of clusters for the given dataset.

# In[288]:


silhouette_scores = []
possible_n_clusters = range(2, 15)
for n_clusters in possible_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    silhouette = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(silhouette)


# This code calculates the Silhouette score for a range of K values.
# The Silhouette is a measure of the quality of a partition of dataset and it is computed by the
# difference between the average distance between a point and points in the same cluster, and the average distance
# between our point and points in near different clusters, it ranges from -1 to 1.
# This approach helps in determining the optimal number of clusters for the given dataset by identifying the K value that maximizes the Silhouette score. The optimal K value will have the highest Silhouette score, indicating the best clustering performance.

# In[298]:


plt.plot(possible_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Numero di cluster')
plt.ylabel('Silhouette')
plt.show()


# This code calculates the Silhouette score for a K-means clustering algorithm with 7 clusters. 
# The "n_clusters" parameter specifies the number of clusters to be formed.
# The "labels_" attribute of the kmeans object returns the cluster labels for each data point.

# In[299]:


kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(X)
silhouette = silhouette_score(X, kmeans.labels_)
print('Silhouette score:', silhouette)


# The Silhouette score is about 0.67, this is a relatively high value that indicating that the clusters are well-separated and compact, with a clear boundary between them.

# In[300]:


X_new= X.iloc[:, :].values
labels= kmeans.fit_predict(X)
print(labels)


# The output above represents the cluster labels assigned to each observation. 
# In this case, each observation is assigned a label from 0 to 6, indicating which of the 7 identified clusters it belongs to.

# In[301]:


plt.scatter(X_new[:, 0], X_new[:, 1], c=labels, s=50, cmap='viridis')
plt.show()


# In the plot above, the color of each point is determined by the "labels".
# The "s" parameter controls the size of the plotted points, and the "cmap" parameter specifies the color map.

# In[302]:


##HIERARCHICAL- WARD METHOD
import scipy.cluster.hierarchy as hy
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# The hierarchical methods create sequences of partitions as homogeneous as possibile within and as heretogeneous as possible betewwn. Here we use an agglomerative clustering algorithm that merges clusters according to the ward method linkage.
# Ward aims to group clusters that minimises the sum of squared error. 
# We apply the method and produce a dendrogram.

# In[303]:


Z = linkage(X, 'ward')
fig = plt.figure(figsize=(15, 15))
dn = dendrogram(Z)
plt.title('Ward Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


# From this dendrogram we can note that the best paritition seems to be k=9 clusters.
