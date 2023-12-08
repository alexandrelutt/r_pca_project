import networkx as nx
import numpy as np

class Graph_Laplacian():
  def __init__(self, task = "occlusion"):
    self.task = task
    self.dataset = None
    self.h = None
    self.w = None

  def load_dataset(self, X, n_classes, n_data_by_class):
    self.X = X
    self.h = X.shape[1]
    self.w = X.shape[2]
    self.n_classes = n_classes
    self.n_data_by_class = n_data_by_class

  def compute_weigh_occulsion(self, X, idx1, idx2, with_occulsion = False, occulsion_details = None, occult_size = None):
    im1 = X[idx1]
    im2 = X[idx2]

    mask = np.ones((self.h,self.w))

    if not with_occulsion :
      return np.sqrt((np.multiply(mask, (im1-im2)**2)).sum() / (np.sum(mask)))


    occulted_indices = list(occulsion_details.keys())

    if occulsion_details.get(idx1) is not None :
      x1, y1 = occulsion_details[idx1]
      mask[x1:x1+occult_size, y1:y1+occult_size] = 0

    if occulsion_details.get(idx2) is not None :
      x2, y2 = occulsion_details[idx2]
      mask[x2:x2+occult_size, y2:y2+occult_size] = 0

    return np.sqrt((np.multiply(mask, (im1-im2)**2)).sum() / (np.sum(mask)))


  def generate_graph(self, occulsion_details = None, occult_size = None, sigma = 0.05):
    if self.task == "occlusion" :
      G = nx.Graph()
      G.add_nodes_from([i for i in range(self.n_classes * self.n_data_by_class)])

      for k in range(self.n_classes):
        euclidean_distance_matrix = np.zeros((self.n_data_by_class,self.n_data_by_class))
        for i in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
          for j in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
            euclidean_distance_matrix[i%self.n_data_by_class,j%self.n_data_by_class] = self.compute_weigh_occulsion(self.X, i, j, True, occulsion_details, occult_size)
        w_min = min(euclidean_distance_matrix[i,j] for i in range(self.n_data_by_class) for j in range(i+1, self.n_data_by_class))
        for i in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
          for j in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
            if i < j :
              weight = np.exp(-(euclidean_distance_matrix[i%self.n_data_by_class,j%self.n_data_by_class] - w_min)**2/(sigma**2))
              G.add_edge(i, j, weight = weight)

    elif self.task == "classification" :
      G = nx.Graph()
      G.add_nodes_from([i for i in range(self.n_classes * self.n_data_by_class)])

      for k in range(self.n_classes):
        euclidean_distance_matrix = np.zeros((self.n_data_by_class,self.n_data_by_class))
        for i in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
          for j in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
            euclidean_distance_matrix[i%self.n_data_by_class,j%self.n_data_by_class] = self.compute_weigh_occulsion(self.X, i, j)
        w_min = min(euclidean_distance_matrix[i,j] for i in range(self.n_data_by_class) for j in range(i+1, self.n_data_by_class))
        for i in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
          for j in range(k*self.n_data_by_class, (k+1)*self.n_data_by_class):
            if i < j :
              weight = np.exp(-(euclidean_distance_matrix[i%self.n_data_by_class,j%self.n_data_by_class] - w_min)**2/(sigma**2))
              G.add_edge(i, j, weight = weight)

    else :
      return NotImplementedError

    return G