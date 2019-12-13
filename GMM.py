import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from sklearn.cluster import KMeans


class GMM:
    """
    Implementation of Guassian Mixture Models.
    """
    
    def __init__(self, k, initial_mus=None, initial_covs=None, verbose=0):
        self.k = k
        assert(k >= 1)
        assert(len(initial_mus) == k)
        assert(len(initial_covs) == k)
        self.cluster_centers_ = self.means = initial_mus
        self.covs = initial_covs
        self.verbose = verbose

        # Initialize pis to be equal across the board.
        self.pis = np.array([1/self.k]*self.k)
        
    def v_print(self, content, **kwargs):
        if self.verbose > 0:
            print(content)
    
    def fit(self, X, iters=5):        
        self.mu_history = {k : {"mean": [self.means[k]], 
                                "cov": [self.covs[k]]} for k in range(self.k)}
        self.mu_history["pis"] = [self.pis]
        
        for i in range(iters):
            self.v_print("Iteration: {}".format(i))
            self.v_print("Means: {}".format([m for m in self.means]))
               
            # E Step!
            ynks = np.array([self.ynk(k, X) for k in range(self.k)])
            
            # M Step!
            N_ks = np.array([np.sum(ynks[k], axis=0) for k in range(self.k)])
            new_pis = N_ks/np.sum(N_ks)
            new_mus = np.divide((ynks@X).T, N_ks).T    
            new_covs = []
            for k in range(self.k):
                mean_diff = (X-new_mus[k])[:, np.newaxis]
                cov = np.sum((mean_diff.swapaxes(1,2)@mean_diff)*ynks[k].reshape(len(X), 1, 1), axis=0)
                new_covs.append(cov/N_ks[k])

                
            # Apply the new update/s.
            self.pis = new_pis
            self.mu_history['pis'].append(self.pis)
            for k in range(self.k):
                self.means[k] = new_mus[k]
                self.mu_history[k]['mean'].append(self.means[k])
                self.covs[k] = new_covs[k]
                self.mu_history[k]['cov'].append(self.covs[k])
    
    def ynk(self, k, x):
        """
        Calculates p(z=k|X, theta)
        """
        ynks = [self.pis[j]*multivariate_normal.pdf(x, mean=self.means[j], cov=self.covs[j]) for j in range(self.k)]
        return np.nan_to_num(ynks[k]/np.sum(ynks, axis=0))
        
    def display_gmm(self, X, ax, ix=-1, mean_prog=False):
        """
        Nice helper method to generate beautiful plots of guassian distributions.
        """
        # Set given index to the current guassians
        current_means = self.means.copy()
        current_covs = self.covs.copy()
        current_pis = self.pis.copy()
        
        self.means = [self.mu_history[k]["mean"][ix] for k in range(self.k)]
        self.covs = [self.mu_history[k]["cov"][ix] for k in range(self.k)]
        self.pis = self.mu_history["pis"][ix]
        
        # Make predictions.
        preds = self.predict(X)

        # Plot the data.
        ax.scatter(X[:, 0], X[:, 1], c=preds)
        ax.scatter(X[:, 0], X[:, 1], c=preds)
        
        # Draw the progression of the mean if specified.
        if mean_prog:
            # Display the mean progression.
            for k in range(self.k):
                ax.plot(*np.array(self.mu_history[k]['mean']).T)

        # Draw the gaussian distributions.
        for k in range(self.k):
            self.draw_multivariate(multivariate_normal(mean=self.means[k], cov=self.covs[k]), ax, alpha=.2)            
        # Plot the mean points.
        for k in range(self.k):
            ax.scatter(*self.means[k], s=200)
            
        # Return the values to their correct values.
        self.means = current_means
        self.covs = current_covs
        self.pis = current_pis
    
    # Modified method based on one seen in https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    def draw_multivariate(self, multi_norm, ax, **kwargs):
        # Convert covariance to principal axes
        U, s, Vt = np.linalg.svd(multi_norm.cov)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)

        # Draw the Ellipse
        for nsig in range(1, 3):
            ax.add_patch(Ellipse(multi_norm.mean, nsig * width, nsig * height, angle, **kwargs))
            
    def predict(self, X):
        """
        Calculates probability of being in cluster and takes maximum for each data point.
        """
        return np.argmax(np.array([self.ynk(k, X) for k in range(self.k)]).T, axis=1)
    
    def probs(self, X):
        return np.array([self.ynk(k, X) for k in range(self.k)]).T
    
    def __str__(self):
        return "Gaussian Mixture Model k: {}".format(self.k)
        
    def __repr__(self):
        return "GMM(k={})".format(self.k)