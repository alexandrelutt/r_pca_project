import numpy as np
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye


def one_norm(X):
    return np.sum(np.abs(X))

def froben_norm(X):
    return np.linalg.norm(X, ord='fro')

class RobustPCA():
    def __init__(self, lambd=None, mu=None, max_iter=100, tol=1e-6):
        self.lambd = lambd
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol

    def error(self, L, S):
        return froben_norm(self.M - L - S)

    def D_op(self, X):
        U, S, V = np.linalg.svd(X, full_matrices=False)
        tau = self.inv_mu
        S = S[S>tau] - tau
        rank = len(S)
        return np.dot(U[:, :rank]*S, V[:rank,:])

    def S_op(self, tau, X):
        return np.sign(X)*np.maximum(np.abs(X)-tau, 0)

    def fit(self, data):
        self.M = data

        if not self.mu:
            self.mu = self.M.shape[0]*self.M.shape[1]/(4*one_norm(self.M))
        self.inv_mu = 1/self.mu

        if not self.lambd:
            self.lambd = 1/np.sqrt(np.max(self.M.shape))

        S = np.zeros(self.M.shape)
        Y = np.zeros(self.M.shape)

        for _ in range(self.max_iter):
            L = self.D_op(self.M - S + self.inv_mu*Y)
            S = self.S_op(self.lambd*self.inv_mu, self.M - L + self.inv_mu*Y)

            self.err = self.error(L, S)
            if self.err < self.tol:
                break

            Y += self.mu*(self.M - L - S)

        return L, S

class GLPCA():
  def __init__(self, beta, k):
    # k : dimension of the projective space
    # beta : trade-off between the two terms of the objective function (see the article of the GLPCA)
    self.beta = beta
    self.k = k

  def fit(self, X, graph):
    h, w = X[0].shape
    X = X.reshape(len(X), h*w).T

    nb_nodes = graph.number_of_nodes()

    W = nx.adjacency_matrix(graph)
    degree_sequence = np.array([graph.degree(node) for node in graph.nodes()])
    D = diags(degree_sequence)
    L = D - W

    XTX = np.matmul(X.T, X)
    A = XTX / eigs(XTX, k=1, which = "LM")[0].real

    B = L / eigs(L, k=1, which = "LM")[0].real
    G = (1-self.beta)*(eye(nb_nodes) - A) + self.beta*B

    eigenvalues, eigenvectors = eigs(G, k=self.k, which = "SM")
    Q = np.real(eigenvectors)
    U = X@Q
    return Q, U

def get_model(model_name):
    if model_name == 'RobustPCA':
        new_model = RobustPCA()
    else:
        raise NotImplementedError
    return new_model