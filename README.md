# Age-and-Income-Clustering-with-KMeans
This project demonstrates how to perform KMeans clustering on a dataset containing age and income information. It visualizes the clusters and centroids and applies preprocessing techniques such as scaling and label encoding.

Project Overview
This project aims to:

Apply KMeans clustering on a dataset to find patterns between age and income.
Visualize the clusters with different colors.
Use preprocessing steps like scaling and encoding for optimal clustering performance.

Dataset
The dataset used contains two main features:

Age - The age of individuals.
Income($) - The annual income in dollars.

Key Libraries Used
Numpy
Pandas
Matplotlib
Scikit-learn
Steps Performed
Data Loading and Visualization
We loaded the data and created a scatter plot of age vs. income to get an initial visual of the dataset.

KMeans Clustering
We applied KMeans clustering with 3 clusters, then visualized the clusters using different colors.

Data Preprocessing

Used MinMaxScaler to normalize the age and income data.
Applied LabelEncoder for transforming the age and income data into numerical labels.
Reapplying KMeans Clustering
After preprocessing, KMeans clustering was re-applied to the transformed data and clusters were visualized again.

How to Run the Project
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/age-income-clustering.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the script:

bash
Copy code
python kmeans_clustering.py
Results
The final plot shows three distinct clusters of individuals based on age and income, with cluster centroids highlighted.

Future Enhancements
Add more features to the dataset for better clustering.
Explore different clustering algorithms (e.g., DBSCAN, Hierarchical Clustering).
Evaluate clustering performance with metrics like Silhouette Score.

