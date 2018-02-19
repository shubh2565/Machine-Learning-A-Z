import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




#loading of dataset
data = pd.read_csv('Mall_Customers.csv')
print(data)
X = data.iloc[ : , 3:].values
print('\n{}'.format(X))


#using elbow method to find optimal number of clusters
wcss = []
for i in range(1,11):
	kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


#fitting K-means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(X)


#visualizing the cluster
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c='red', label='Cautious')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c='grey', label='Standard')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c='blue', label='Target')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c='violet', label='Careless')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c='pink', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()