import numpy as np 
import matplotlib.pyplot as plt

import cv2

W, H = 128, 128
occlusion_factor = 0.1
dx, dy = int(np.sqrt(occlusion_factor)*W), int(np.sqrt(occlusion_factor)*H)
number_list = [f"{i:03d}" for i in range(251)]

def load_img(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img

def display(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def corrupt(img):
    new_img = img.copy()
    start_x, start_y = np.random.randint(0, W-dx), np.random.randint(0, H-dy)
    end_x, end_y = start_x + dx, start_y + dy
    new_img[start_x:end_x, start_y:end_y] = 0
    return new_img

def compare(img1, img2, img3, model_name, corrupted, save, i):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    base_img = 'Base'
    if corrupted:
        base_img = 'Corrupted base'

    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(f'{base_img} image')
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Retrieved image')
    axes[1].axis('off')

    axes[2].imshow(img3, cmap='gray')
    axes[2].set_title('Sparse component')
    axes[2].axis('off')
    
    if save:
        if corrupted:
            save_filename = f'figures/{model_name}_unmasking_example_{i}.png'
        else:
            save_filename = f'figures/{model_name}_shadow_removing_example_{i}.png'
            
        plt.savefig(save_filename)
    plt.show()