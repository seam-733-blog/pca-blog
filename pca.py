# Import Statements
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# PCA Algorithm Implementation
def PCA(scaled_data, n=2):
    
    # Calculating the covariance matrix
    covariance_matrix = np.cov(scaled_data.T)
    
    # Computing eigenvalues, eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sorting the eigenvectors according to eigenvalues
    eigenvectors = eigenvectors.T
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[sorted_indices]
    
    # Get the first n eigenvectors
    eigen_pca = eigenvectors[0:n]
    
    # Prinicipal Components
    pca_values = np.dot(scaled_data, eigen_pca.T)
    
    return pca_values


# Main Logic
if __name__ == '__main__':
    
    # Load the Sample Dataset
    dataset = load_breast_cancer()
    sample_df = pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
    
    #Performing Standardization
    standard_df = (sample_df - sample_df.mean()) / sample_df.std()
    standard_df.head(10)
    
    # Applying PCA to the dataset
    principal_comps = PCA(standard_df, n=2)
    plt.scatter(principal_comps[:, 0], principal_comps[:, 1], c=dataset['target'], cmap="plasma")
