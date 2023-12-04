from source_code.evaluation import *

def plot_dataset(X, h, w, n_data_by_class, title):
    n_rows = int(np.ceil(n_data_by_class/10))

    fig, axs = plt.subplots(n_rows, 10)
    fig.set_size_inches(20, 2.5*n_rows)
    fig.suptitle(title, fontsize=25)
    for i in range(n_data_by_class):
        if n_rows == 1 :
            axs[i].imshow(X[i].reshape(h,w), cmap = "gray")
            axs[i].axis('off')
        else :
            axs[int(np.floor(i/10)), i%10].imshow(X[i].reshape(h,w), cmap = "gray")
            axs[int(np.floor(i/10)), i%10].axis('off')
    if n_data_by_class%10 != 0 :
        for i in range(n_data_by_class%10, 10) :
            axs[n_rows-1, i].axis('off')

def compare_methods(dataset, n_data_by_class, beta, seed = 42) :
    np.random.seed(seed)
    n_occult = n_data_by_class

    X_data = load_dataset(dataset, n_data_by_class, "random")
    h, w = X_data.shape[1], X_data.shape[2]

    X_occulted, G = occult_and_generate_graph(X_data, dataset, 25, n_occult, n_data_by_class)
    print(X_occulted.shape)
    print(G.number_of_nodes())
    h, w = X_occulted.shape[1], X_occulted.shape[2]
    # RPCA
    L_RPCA, S_RPCA = evaluate_RPCA(X_occulted, n_data_by_class = n_data_by_class, plot = False)
    # GLPCA
    X_GLPCA_by_beta = evaluate_GLPCA(X_occulted, G, beta_vals = [0, beta], n_data_by_class = n_data_by_class, plot = False)
    # RGLPCA
    X_RGLPCA_by_beta = evaluate_RGLPCA(X_occulted, G, beta_vals = [beta], n_data_by_class = n_data_by_class, plot = False)
    # OURPCA
    L_OURPCA, S_OURPCA = evaluate_OURPCA(X_occulted, G, n_data_by_class = n_data_by_class, plot = False)

    # Plot
    # Original dataset
    plot_dataset(X_data, h, w, n_data_by_class, title = "Original dataset")
    # Occulted dataset
    plot_dataset(X_occulted, h, w, n_data_by_class, title = "Occulted dataset")
    # PCA (beta = 0 in GLPCA)
    plot_dataset(X_GLPCA_by_beta[0], h, w, n_data_by_class, title = "PCA")
    # RPCA
    plot_dataset(L_RPCA, h, w, n_data_by_class, title = "RPCA")
    # GLPCA
    plot_dataset(X_GLPCA_by_beta[beta], h, w, n_data_by_class, title = r"GLPCA; $\beta$ = " + str(beta))
    # RGLPCA
    plot_dataset(X_RGLPCA_by_beta[beta], h, w, n_data_by_class, title = r"RGLPCA; $\beta$ = " + str(beta))
    # OURPCA (RPCA on Graph)
    plot_dataset(L_OURPCA, h, w, n_data_by_class, title = "RPCA on Graphs")

    plt.show()