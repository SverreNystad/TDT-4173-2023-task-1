import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class KMeans:
    
    def __init__(self, k: int=2, init_method: str="random", max_iteration=10):
        self.k = k # number of clusters
        self.centroids = np.zeros(k) # cluster centroids

        self.init_method = init_method # method to use for initialization
        self.max_iteration = max_iteration # maximum number of iterations to run the algorithm
        
    def fit(self, X: pd.DataFrame):
        """
        Estimates parameters for the classifier
        The fit method implements the naïve k-means clustering algorithm.
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        print(f"self.init_method = {self.init_method}, self.k = {self.k}, X.shape = {X.shape}")
        # Initialize cluster centroids
        self.centroids = initiate_cluster_centroids(X, self.k, self.init_method)

        clusters = np.zeros(len(X))
        for iteration in range(0, self.max_iteration):
            # Assign each point to the closest centroid
            clusters = self.predict(X)

            # Update the centroids based on the new cluster assignments
            # Find the new centroids by taking the average value
            for centroid_name, centroid in enumerate(self.centroids):
                # Get the points that belong to the centroid
                points = []
                for i, cluster in enumerate(clusters):
                    if cluster == centroid_name:
                        points.append(X.iloc[i])
                
                # Calculate the average value of the points
                self.centroids[centroid_name] = np.average(points, axis=0)
    

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        samples = np.shape(X)[0]
        # Make predictions for each sample and as they will be used as indices, they need to be integers
        predictions = np.zeros(samples, dtype=np.int)


        for sample in range(0, samples):
            # Find the closest centroid
            closest_centroid = 0
            closest_distance = np.inf
            for centroid_name, centroid in enumerate(self.centroids):
                distance = euclidean_distance(X.iloc[sample].values, centroid)
                if distance < closest_distance:
                    closest_centroid = centroid_name
                    closest_distance = distance
            # Make the prediction feature-wise corresponding to the closest centroid feature-wise
            predictions[sample] = closest_centroid
        return predictions


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


def feature_engineering(X: pd.DataFrame, method="normalize") -> pd.DataFrame:
    """
    Feature engineering for the data
    """
    if method == "normalize":
        return normalize_data(X)
    if method == "standardize":
        return standardize_data(X)
    else:
        raise ValueError(f"Unknown method: {method}, valid methods are: 'normalize' and 'standardize'")


def normalize_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the data to be between 0 and 1
    """
    # Shift the data so that the minimum value is 0
    x_shift = (X-X.min()) 
    # Scale the data so that the maximum value is 1
    feature_range = (X.max()-X.min())
    return x_shift / feature_range

def standardize_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the data to have mean 0 and standard deviation 1
    """
    # Shift the data so that the mean is 0
    x_shift = (X-X.mean())
    # Scale the data so that the standard deviation is 1
    return x_shift / X.std()

def initiate_cluster_centroids(X: pd.DataFrame, k: int, method="random") -> np.ndarray:
    """
    Initializes the cluster centroids
    
    Args:
        X (array<m,n>): a matrix of floats with
            m rows (#samples) and n columns (#features)
        k (int): number of clusters
        method (str): method to use for initialization,
            valid methods are: 'random', 'kmeans++'
    """
    if method == "random":
        return random_initiate_cluster_centroids(X, k)
    elif method == "kmeans++":
        return kmeans_plus_plus_initiate_cluster_centroids(X, k)
    else:
        raise ValueError(f"Unknown method: {method}, valid methods are: 'random', 'kmeans++'")

def random_initiate_cluster_centroids(X: pd.DataFrame, k: int) -> np.ndarray:
    """
    Initializes the cluster centroids by using random data points.
    """
    # Initialize the centroids to be random points
    dimension = X.shape[1]
    centroids = np.zeros((k, dimension))
    # Choose one center uniformly at random from among the data points.
    for i in range(0, k):
        random_row = np.random.randint(X.shape[0])
        choice = X.iloc[random_row]
        centroids[i] = choice

    return centroids

def kmeans_plus_plus_initiate_cluster_centroids(X: pd.DataFrame, k: int) -> np.ndarray:
    """
    Initializes the cluster centroids by using k-means++
    K-means++ is a smart way to initialize the centroids by first selecting a random point and then choosing the next points to be spread out as much as possible from the previous ones.
    Learn more at: https://en.wikipedia.org/wiki/K-means%2B%2B
    Args:
        X (array<m,n>): a matrix of floats with
            m rows (#samples) and n columns (#features)
        k (int): number of clusters
        
    Returns:
        A k x n float matrix with the centroids
    """
    # Choose one center uniformly at random from among the data points.
    clusters = np.zeros((k, X.shape[1]))
    # Chose the first at random
    clusters[0] = X[np.random.choice(len(X))]

    # For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
    for i in range(1, k):
        # Compute the distance between each point and the nearest center
        distance = np.zeros(len(X))
        for j in range(0, len(X)):
            distance[j] = min(euclidean_distance(X[j], clusters[0:i]))
        # Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
        probability = distance**2/np.sum(distance**2)
        index_of_new_center = np.random.choice(len(X), p=probability)
        clusters[i] = X[index_of_new_center]

    return clusters


# --- Some utility functions 

def euclidean_distance(x, y) -> np.ndarray:
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None) -> np.ndarray:
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z) -> float:
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for _, cluster in enumerate(clusters):
        points_in_cluster = X[z == cluster] # Xc = points_in_cluster
        # mean_point is the centroid of the cluster
        centroid = points_in_cluster.mean(axis=0) # mu = centroid
        distortion += ((points_in_cluster - centroid) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z) -> float:
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    Silhouette measures how well object i matches the clustering at hand (that is, how well
    it has been classified).
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, cluster_a in enumerate(clusters):
        for j, cluster_b in enumerate(clusters):
            in_cluster_a = z == cluster_a
            in_cluster_b = z == cluster_b
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum() / np.clip(div, 1, None)
    
    # Intra distance 
    # a is the dissimilarity of a point in a cluster to the other objects within its cluster.
    a = D[np.arange(len(X)), z] 
    
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    # The silhouette score is the difference between the average intra-cluster distance and the smallest inter-cluster distance divided by the maximum of the two.
    # Find the average silhouette score for the dataset by computing the silhouette score for each point, then taking the average.
    return np.mean((b - a) / np.maximum(a, b))
  