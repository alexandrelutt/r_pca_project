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

class RGLPCA():
    def __init__(self, beta, k, rho = 1.2, n_iter = 10):
        # k : dimension of the projective space
        # beta : trade-off between the two terms of the objective function (see the article of the GLPCA)
        self.beta = beta
        self.k = k
        self.rho = rho
        self.n_iter = n_iter

    def fit(self, X, graph):
        beta_init = self.beta
        X_init = X.copy()

        h, w = X[0].shape

        #Init E :
        E = np.ones(X.shape)
        #Init C and mu :
        C = np.zeros(X.shape)
        mu = 1

        W = nx.adjacency_matrix(graph)
        degree_sequence = np.array([graph.degree(node) for node in graph.nodes()])
        D = diags(degree_sequence)
        L = D - W

        xi = eigs(L, k=1, which = "LM")[0].real

        for iter in range(self.n_iter):
            X = X_init - E - C/mu
            X1 = X.reshape(len(X), h*w).T
            XTX = np.matmul(X1.T, X1)
            lmbda = eigs(XTX, k=1, which = "LM")[0].real

            alpha = beta_init/(1 - beta_init)*lmbda/xi
            alpha = 2*alpha/mu
            beta = alpha*xi/(lmbda + alpha*xi)[0]

            # Solve E subproblem

            # Solve Q, U with E fixed
            Q, U = GLPCA(beta[0], self.k).fit(X, graph)

            # Solve E with Q, U fixed
            UQT = (U@Q.T).reshape(len(X), h, w)
            A = X - UQT - C/mu
            A = A.reshape(len(X), h*w).T
            a = (A**2).sum(axis=1) # a is the vector of the norms of the rows of A
            a = np.sqrt(a)
            E = np.multiply(np.maximum(1 - 1/(mu*a), 0)[:, None], A).T
            E = E.reshape(len(X), h, w)

            # Parameters update
            C = C + mu*(E - X + UQT)
            mu = mu*self.rho

        return Q, U, E

class OurPCA():
    def __init__(self, max_iter=100, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol

    def get_phi(self, G):
        A = nx.adjacency_matrix(G).toarray()
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        phi = D_inv_sqrt @ A @ D_inv_sqrt
        return phi

    def omega_operator(self, tau, A):
        return np.sign(A)*np.maximum(np.abs(A)-tau, 0)

    def D_operator(self, tau, A):
        P, sigma, Qh = np.linalg.svd(A, full_matrices=False)
        return P @ self.omega_operator(tau, np.diag(sigma)) @ Qh

    def has_converged(self, P, new_P, verbose=False):
        err = np.linalg.norm(new_P - P)/np.linalg.norm(P)
        if verbose:
            print(err)
        return err < self.tol

    def train(self, X, phi, gamma):
        p, n = X.shape

        lambd = 1/np.sqrt(np.max(X.shape))

        L = np.random.random((p, n))
        W = np.random.random((p, n))
        S = np.random.random((p, n))

        r_1, r_2 = 1, 1

        Z_1 = X - L - S
        Z_2 = W - L

        for t in range(self.max_iter):
            H_1 = X - S + Z_1/r_1
            H_2 = W + Z_2/r_2
            A = (r_1*H_1 + r_2*H_2)/(r_1 + r_2)
            r = (r_1 + r_2)/2
            L = self.D_operator(1/r, A)

            S = self.omega_operator(lambd/r_1, X - L + Z_1/r_1)
            W = r_2*np.linalg.inv(gamma*phi + r_2*np.identity(p)) @ (L - Z_2/r_2)

            Z_1 = Z_1 + r_1*(X - L - S)
            Z_2 = Z_2 + r_2*(W - L)

        return L, S
    
    def fit(self, X, G, gamma):
        phi = self.get_phi(G)
        L, S = self.train(X, phi, gamma)
        return L, S