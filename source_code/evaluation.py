from source_code.models import *
from source_code.tools import *

possible_methods = ['RPCA', 'GLPCA', 'RGLPCA']
possible_tasks = ['Unmasking', 'Shadow removing', 'Recovering artificial dataset']

def plot_random_exposures(X, L, S, model_name, corrupted, save):
    ids = np.random.randint(0, X.shape[1], 3)
    for i in ids:
        img_X = X[:, i].reshape(W, H)
        img_L = L[:, i].reshape(W, H)
        img_S = S[:, i].reshape(W, H)
        compare(img_X, img_L, img_S, model_name, corrupted, save, i)

def load_pie_dataset(corrupted):
    image_idx = np.random.choice(number_list)
    imgs = []
    for j in [0, 1]:
        for i in range(10):
            filename = f'small_PIE_dataset/{image_idx}/{image_idx}_01_01_051_{j}{i}_crop_128.png'
            img = load_img(filename)
            if corrupted:
                img = corrupt(img)
            flat_img = img.flatten()
            imgs.append(flat_img)
    X = np.array(imgs).T
    return X

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
        X = load_pie_dataset(corrupted=corrupted)

        print(f'Training {model_name} model for {task.lower()}...')
        model = get_model(model_name)
        L, S = model.fit(X)

        plot_random_exposures(X, L, S, model_name=model_name, corrupted=corrupted, save=save)

    else:
        display_retrieval_efficiency(model_name, choosen_scheme, save=save)     