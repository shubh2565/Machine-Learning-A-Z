import pandas as pd 
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering



#loading of dataset
data = pd.read_csv('Mall_Customers.csv')
print(data)
X = data.iloc[ : , 3:].values
print('\n{}'.format(X))


#using dendrogram to find optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()


#fitting hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)


#visualization
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c='red', label='Cautious')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c='grey', label='Standard')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c='blue', label='Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c='violet', label='Careless')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c='pink', label='Sensible')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()