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

def evaluate_GLPCA(task = "Unmasking", dataset = "att", save = False, beta_vals = [0, 0.3, 0.5, 1],\
                    k = 3, occult_percent = 25):

    assert dataset in ["att", "pie"]

    if dataset == "att" :
        print("Loading the AT&T dataset...")
        idx = np.random.randint(0,40)
        X_data, h, w = load_att_dataset()
        X_data = X_data[idx*10:(idx+1)*10]
        n_data_by_class = 10

    if dataset == "pie" :
        print("Loading the PIE dataset...")
        X_data, h, w = load_pie_dataset()
        X_data = X_data[:10]
        n_data_by_class = 10

    if task == "Unmasking" :
        occult_size = int(occult_percent/100 * min(h,w))
        X_occulted, occulsion_details = occult_dataset(X_data, occult_size, n_occult=2)

        G_laplacian = Graph_Laplacian()
        G_laplacian.load_dataset(X_occulted, 1, n_data_by_class)
        G = G_laplacian.generate_graph(occulsion_details, occult_size)

        X_PCA_by_beta = dict()

        for beta in beta_vals:
            GlPCA_model = GLPCA(beta = beta, k = k)
            Q, U = GlPCA_model.fit(X_occulted, G)
            X_PCA_by_beta[beta] = ((U@Q.T).T).reshape(1*n_data_by_class,h,w)

        nb_rows = len(beta_vals) + 1

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

    if save :
        fig.savefig("./figures/evaluation_GLPCA.png")
    plt.show()

def evaluate_RGLPCA(task = "Unmasking", dataset = "att", save = False, beta_vals = [0, 0.3, 0.5],\
                    k = 3, occult_percent = 25) :

    assert dataset in ["att", "pie"]

    if dataset == "att" :
        print("Loading the AT&T dataset...")
        idx = np.random.randint(0,40)
        X_data, h, w = load_att_dataset()
        X_data = X_data[idx*10:(idx+1)*10]
        n_data_by_class = 10

    if dataset == "pie" :
        print("Loading the PIE dataset...")
        X_data, h, w = load_pie_dataset()
        X_data = X_data[:10]
        n_data_by_class = 10

    if task == "Unmasking" :
        occult_size = int(occult_percent/100 * min(h,w))
        X_occulted, occulsion_details = occult_dataset(X_data, occult_size, n_occult=2)

        G_laplacian = Graph_Laplacian()
        G_laplacian.load_dataset(X_occulted, 1, n_data_by_class)
        G = G_laplacian.generate_graph(occulsion_details, occult_size)

        X_PCA_by_beta = dict()

        for beta in beta_vals:
            RGlPCA_model = RGLPCA(beta = beta, k = k)
            Q, U, E = RGlPCA_model.fit(X_occulted, G)
            X_PCA_by_beta[beta] = ((U@Q.T).T).reshape(1*n_data_by_class,h,w)

        nb_rows = len(beta_vals) + 1

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

    if save:
        fig.savefig("./figures/evaluation_RGLPCA.png")
    plt.show()

def evaluate_OURPCA(task = "Unmasking", dataset = "pie", gamma=1e-3, save = False, occult_percent = 25):

    assert dataset in ["att", "pie"]

    if dataset == "att" :
        print("Loading the AT&T dataset...")
        idx = np.random.randint(0,40)
        X_data, h, w = load_att_dataset()
        X_data = X_data[idx*10:(idx+1)*10]
        n_data_by_class = len(X_data)

    if dataset == "pie" :
        print("Loading the PIE dataset...")
        X_data, h, w = load_pie_dataset()
        n_data_by_class = len(X_data)

    if task == "Unmasking" :
        occult_size = int(occult_percent/100 * min(h,w))
        X_occulted, occulsion_details = occult_dataset(X_data, occult_size)

        G_laplacian = Graph_Laplacian()
        G_laplacian.load_dataset(X_occulted, 1, n_data_by_class)
        G = G_laplacian.generate_graph(occulsion_details, occult_size)
        X_occulted = X_occulted.reshape(X_occulted.shape[0], h*w)
        
        OURPCA_model = OurPCA()
        L, S = OURPCA_model.fit(X_occulted, G, gamma=gamma)
        plot_random_exposures(X_data.reshape(-1, h, w), L.reshape(-1, h, w), S.reshape(-1, h, w), model_name='Our model', corrupted=True, save=save)
