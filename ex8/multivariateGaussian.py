#MULTIVARIATEGAUSSIAN Computes the probability density function of the
#multivariate gaussian distribution.
#    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
#    density function of the examples X under the multivariate gaussian
#    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
#    treated as the covariance matrix. If Sigma2 is a vector, it is treated
#    as the \sigma^2 values of the variances in each dimension (a diagonal
#    covariance matrix)
#
import numpy as np

def multivariateGaussian(X, mu, sigma2):
    k = len(mu)

    if sigma2.ndim == 1:
        sigma2M = np.diag(sigma2)
    else:
        sigma2M = sigma2

    sigma_inv = np.linalg.pinv(sigma2M)

    p = (2 * np.pi) ** (- k/2) * np.linalg.det(sigma2M) ** (-1/2) * \
            np.exp(-1/2 * np.sum( (X-mu).dot(sigma_inv)*(X-mu), axis =1))

    return p
