from scipy.ndimage import rotate
import cv2
import glob
import numpy as np
import os


def transform_into_binary(image_path, new_image_path):
    # Charger l'image d'origine en noir et blanc
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Enlever le vignetting
    height, width = img.shape
    kernel_x = cv2.getGaussianKernel(width, 750)
    kernel_y = cv2.getGaussianKernel(height, 750)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask += 1-mask.max()
    vignette = img/mask
    vignette[vignette>255] = 255

    # Ajuster le gamma pour augmenter le contraste
    gamma = 255*(vignette/255)**(10)
    gamma = gamma.astype(np.uint8)
    contr = cv2.normalize(gamma, None, 0, 255, cv2.NORM_MINMAX)

    # Appliquer la méthode d'Otsu pour trouver l'objet
    _, th = cv2.threshold(contr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverser le canal S pour les parties sombres/noires
    th = cv2.bitwise_not(th)

    # Appliquer la fermeture morphologique pour combler les espaces entre les carrés
    kernel = np.ones((5, 5), np.uint8)
    closed_th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # Appliquer une fermeture morphologique pour combler les petits espaces
    # Utiliser un noyau plus grand pour combler des trous plus grands
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(closed_th, cv2.MORPH_CLOSE, kernel)

    # Sauvegarder l'image résultante
    cv2.imwrite(new_image_path, closed)
    #print("L'image avec la plus grande figure blanche sur fond noir a été sauvegardée avec succès.")
    return True

def laminogram(sinogram):
    # Calculer les paramètres à partir du sinograme
    nbprj, nbpix = sinogram.shape
    
    # Initialiser une image vide
    image = np.ones((nbpix, nbpix))

    # Comme la deuxième moitié des projections est symétrique,
    # on peut traiter de seulement la première moitié
    for i in range(nbprj):
        # Répéter la projection sur plusieurs colonnes
        # pour l'étaler  sur toute l'image à reconstruire
        sinogram_image = np.tile(sinogram[i], (nbpix, 1))

        # Tourne la projection étalée selon l'angle à laquelle elle a été prise
        rotated_list = rotate(sinogram_image, 360*i/nbprj, reshape=False, order=0)

        # Ajouter les listes tournées
        image *= rotated_list
    
    return image


# Chemin du dossier contenant les photos de Rubik's cubes
input_folder = 'Images' # a changer suivant le dossier des photos
output_folder = 'Images_binaire' # nom dossier phiotos traitées
recon_folder = 'Images_recon' # nom dossier reconstruction

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parcourir toutes les images dans le dossier
images_binaires = []

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # Charger l'image
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        images_binaires.append(transform_into_binary(input_path, output_path))

images_binaires = np.array(images_binaires)/255

# Singlethreading
"""
for i in range(0, images_binaires.shape[1], 25):
    image_recon = laminogram(images_binaires[:,i,:])
    output_path = os.path.join(recon_folder, f"couche_{i}.JPG")
    cv2.imwrite(output_path, image_recon*255)
"""

# Multithreading
import concurrent.futures


def thread_function(i, sinogram):
    image_recon = laminogram(sinogram)
    output_path = os.path.join(recon_folder, f"couche_{i}.JPG")
    cv2.imwrite(output_path, image_recon*255)


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(thread_function, range(images_binaires.shape[-1]), [images_binaires[:, i, :] for i in range(images_binaires.shape[1])])
