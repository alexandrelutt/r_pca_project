from source_code.models import *
from source_code.tools import *
from source_code.graphs import *
from scipy.sparse import coo_matrix

possible_methods = ['RPCA', 'GLPCA', 'RGLPCA']
possible_tasks = ['Unmasking', 'Shadow removing', 'Recovering artificial dataset']

def plot_random_exposures(X, L, S, model_name, corrupted, save):
    ids = np.random.randint(0, X.shape[0], 3)
    for i in ids:
        img_X = X[i, :, :]
        img_L = L[i, :, :]
        img_S = S[i, :, :]
        compare(img_X, img_L, img_S, model_name, corrupted, save, i)

def load_pie_dataset():
    image_idx = np.random.choice(number_list)
    imgs = []
    for image_idx in number_list:
        for j in [0, 1]:
            for i in range(10):
                filename = f'small_PIE_dataset/{image_idx}/{image_idx}_01_01_051_{j}{i}_crop_128.png'
                img = load_img(filename)
                flat_img = img.flatten()
                imgs.append(flat_img)
    X = np.array(imgs).reshape(-1, 64, 64)
    return X/255, 64, 64

def load_att_dataset():
  h = 112
  w = 92

  X_load = np.zeros((400, h*w))
  for cl in range(1, 41):
    for i in range(1, 11):
      img = "./att_dataset/s" + str(cl) + "/" + str(i) + ".pgm"
      img = np.array(read_pgm(open(img, 'rb')))
      X_load[10*(cl - 1) + (i-1)] = img
  X_load = X_load / 255

  return X_load.reshape(X_load.shape[0], h, w), h, w

def generate_low_rank_matrix(n, d, error_fraction, choosen_cheme):
    A = np.random.randn(d, n)/np.sqrt(n)
    B = np.random.randn(d, n)/np.sqrt(n)

    L = np.dot(A.T, B)
    corrupted_L = L.copy()

    corrupted_mask = np.random.choice([0, 1], size=(n, n), p=[1 - error_fraction, error_fraction])

    if choosen_cheme == 'random':
        corrupted_L[corrupted_mask == 1] = np.random.choice([-1, 1], size=(corrupted_mask == 1).sum())

    elif choosen_cheme == 'coherent':
        sign_matrix = np.sign(L)
        corrupted_L[corrupted_mask == 1] = sign_matrix[corrupted_mask == 1]

    return L, corrupted_L

def compute_error(model_name, d_n, error_fraction, choosen_scheme):
    n = 400
    d = int(d_n * n)
    L, corrupted_L = generate_low_rank_matrix(n, d, error_fraction, choosen_scheme)
    model = get_model(model_name)
    hat_L, halt_S = model.fit(corrupted_L)
    error = np.log10((froben_norm(L-hat_L))/(froben_norm(L)))
    return error

def display_retrieval_efficiency(model_name, choosen_scheme, save):
    assert choosen_scheme in ['random', 'coherent']

    d_ns = np.linspace(0.06, 0.3, 10)
    error_fractions = np.linspace(0.02, 0.3, 10)
    X, Y = np.meshgrid(d_ns, error_fractions)
    print(f'Training {model_name} model for recovering artificial dataset ({choosen_scheme} scheme)...')
    Z = np.array([[compute_error(model_name, d_n, error_fraction, choosen_scheme) for error_fraction in error_fractions] for d_n in d_ns])

    plt.imshow(Z, extent=[d_ns.min(), d_ns.max(), error_fractions.min(), error_fractions.max()], origin='lower',
            cmap='turbo', aspect='auto')
    plt.colorbar()
    plt.xlabel(r'$\frac{1}{n} \times Rank(L)$')
    plt.ylabel(r'$\frac{1}{n^2} \times ||S||_0$')
    plt.title(r'$\log_{10}(||L - \hat{L}||_F/||L||_F)$' + f' ({choosen_scheme} scheme)')
    if save:
        plt.savefig(f'figures/{model_name}_retrieval_efficiency_{choosen_scheme}_scheme.png')
    plt.show()

def evaluate(model_name, task='Unmasking', choosen_scheme='random', save=False):
    if task in ['Unmasking', 'Shadow removing']:
        corrupted = (task == 'Unmasking')
        X, h, w = load_pie_dataset()
        occult_size = int(25/100 * min(h, w))
        X, _ = occult_dataset(X, occult_size)
        print(f'Training {model_name} model for {task.lower()}...')
        model = get_model(model_name)
        L, S = model.fit(X.reshape(X.shape[0], h*w))

        X = X.reshape(-1, 64, 64)
        L = L.reshape(-1, 64, 64)
        S = S.reshape(-1, 64, 64)

        plot_random_exposures(X, L, S, model_name=model_name, corrupted=corrupted, save=save)

    else:
        display_retrieval_efficiency(model_name, choosen_scheme, save=save)

def load_dataset(dataset, n_data_by_class = 10, all_or_random = "random"):
    assert dataset in ["att", "pie"]

    if dataset == "att" :
        assert n_data_by_class <= 10
        print("Loading the AT&T dataset...")
        X_data, h, w = load_att_dataset()
        if all_or_random == "random" :
            idx = np.random.randint(0,40)
            X_data = X_data[idx*n_data_by_class:(idx+1)*n_data_by_class]

    if dataset == "pie" :
        assert n_data_by_class <= 20
        print("Loading the PIE dataset...")
        X_data, h, w = load_pie_dataset()
        if all_or_random == "random" :
            idx = np.random.randint(0,249)
            X_data = X_data[idx*n_data_by_class:(idx+1)*n_data_by_class]

    return X_data


def occult_and_generate_graph(X_data, dataset, occult_percent, n_occult, n_data_by_class = 10, random_or_all = "random"):
    h, w = X_data.shape[1], X_data.shape[2]
    occult_size = int(occult_percent/100 * min(h,w))
    X_occulted, occulsion_details = occult_dataset(X_data, occult_size)

    G_laplacian = Graph_Laplacian()
    if random_or_all == "random" :
        G_laplacian.load_dataset(X_occulted, 1, n_data_by_class)
    elif random_or_all == "all" :
        G_laplacian.load_dataset(X_occulted, n_classes[dataset], n_data_by_class)

    G = G_laplacian.generate_graph(occulsion_details, occult_size)

    return X_occulted, G

def evaluate_GLPCA(X_occulted, G, task = "Unmasking", beta_vals = [0, 0.3, 0.5, 1],\
                    k = 3, n_data_by_class = 10, plot = True, save = False):
    h, w = X_occulted.shape[1], X_occulted.shape[2]

    if task == "Unmasking" :

        X_PCA_by_beta = dict()

        for beta in beta_vals:
            GlPCA_model = GLPCA(beta = beta, k = k)
            Q, U = GlPCA_model.fit(X_occulted, G)
            X_PCA_by_beta[beta] = ((U@Q.T).T).reshape(1*n_data_by_class,h,w)

        nb_rows = len(beta_vals) + 1

        if plot:
            fig, axs = plt.subplots(nb_rows,n_data_by_class, constrained_layout=True)
            fig.set_size_inches(2.5*n_data_by_class, 2.5*nb_rows)
            for i in range(n_data_by_class):
                axs[0,i].imshow(X_occulted[i], cmap = "gray")
                for j, beta in enumerate(beta_vals):
                    axs[j+1,i].imshow(X_PCA_by_beta[beta][i], cmap = "gray")

            # Titles :
            axs[0,n_data_by_class//2].set_title("Original images", fontsize = 40)
            for i in range(1, nb_rows):
                axs[i,n_data_by_class//2].set_title(r"$\beta = $" + str(beta_vals[i-1]), fontsize = 40)
            plt.show()

        if save and plot:
            fig.savefig("./figures/evaluation_GLPCA.png")

        return X_PCA_by_beta

    # if task == "clustering" :
    #     X_PCA_by_beta = dict()

    #     for beta in beta_vals:
    #         GlPCA_model = GLPCA(beta = beta, k = k)
    #         Q, U = GlPCA_model.fit(X_occulted, G)
            


def evaluate_RGLPCA(X_occulted, G, task = "Unmasking", beta_vals = [0, 0.3, 0.5],\
                    k = 3, n_data_by_class = 10, plot = True, save = False) :

    h, w = X_occulted.shape[1], X_occulted.shape[2]

    if task == "Unmasking" :

        X_PCA_by_beta = dict()

        for beta in beta_vals:
            RGlPCA_model = RGLPCA(beta = beta, k = k)
            Q, U, E = RGlPCA_model.fit(X_occulted, G)
            X_PCA_by_beta[beta] = ((U@Q.T).T).reshape(1*n_data_by_class,h,w)

        nb_rows = len(beta_vals) + 1

        if plot :

            fig, axs = plt.subplots(nb_rows,n_data_by_class, constrained_layout=True)
            fig.set_size_inches(1.5*n_data_by_class, 2.5*nb_rows)
            for i in range(n_data_by_class):
                axs[0,i].imshow(X_occulted[i], cmap = "gray")
                for j, beta in enumerate(beta_vals):
                    axs[j+1,i].imshow(X_PCA_by_beta[beta][i], cmap = "gray")

            # Titles :
            axs[0,n_data_by_class//2].set_title("Original images", fontsize = 40)
            for i in range(1, nb_rows):
                axs[i,n_data_by_class//2].set_title(r"$\beta = $" + str(beta_vals[i-1]), fontsize = 40)

    if save and plot:
        fig.savefig("./figures/evaluation_RGLPCA.png")
    if plot : plt.show()
    return X_PCA_by_beta

def evaluate_OURPCA(X_occulted, G, task = "Unmasking", dataset = "pie", gamma=1e-3, \
                    n_data_by_class = 10, plot = True, save = False):
    h, w = X_occulted.shape[1], X_occulted.shape[2]
    X_occulted = X_occulted.reshape(X_occulted.shape[0], h*w)
    if task == "Unmasking" :

        OURPCA_model = OurPCA()
        L, S = OURPCA_model.fit(X_occulted, G, gamma=gamma)

        if plot :
            fig, axs = plt.subplots(2, n_data_by_class)
            fig.set_size_inches(2.5*n_data_by_class, 5)
            for i in range(n_data_by_class):
                axs[0,i].imshow(X_occulted[i].reshape(h,w), cmap = "gray")
                axs[1,i].imshow(L[i].reshape(h,w), cmap = "gray")
            plt.show()
        if save and plot :
            fig.savefig("./figures/evaluation_OURPCA.png")

        return L, S

def evaluate_RPCA(X_occulted, task='Unmasking', choosen_scheme='random', plot = True, save=False, n_data_by_class = 10):
    model_name = 'RobustPCA'
    h, w = X_occulted.shape[1], X_occulted.shape[2]

    if task in ['Unmasking', 'Shadow removing']:
        corrupted = (task == 'Unmasking')

        print(f'Training {model_name} model for {task.lower()}...')
        model = get_model(model_name)
        L, S = model.fit(X_occulted.reshape(X_occulted.shape[0], h*w))
        if plot :
            fig, axs = plt.subplots(2, n_data_by_class)
            fig.set_size_inches(2.5*n_data_by_class, 5)
            for i in range(n_data_by_class):
                axs[0,i].imshow(X_occulted[i].reshape(h,w), cmap = "gray")
                axs[1,i].imshow(L[i].reshape(h,w), cmap = "gray")
            plt.show()
        if save and plot :
            fig.savefig("./figures/evaluation_RPCA.png")
        # plot_random_exposures(X, L, S, model_name=model_name, corrupted=corrupted, save=save)

    else:
        display_retrieval_efficiency(model_name, choosen_scheme, save=save)

    return L, S