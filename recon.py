from scipy.ndimage import rotate
import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def transform_into_binary(image_path, new_image_path):
    # Charger l'image d'origine en noir et en blanc
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
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # Appliquer une fermeture morphologique pour combler les petits espaces
    # Utiliser un noyau plus grand pour combler des trous plus grands
    #kernel = np.ones((15, 15), np.uint8)
    #closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

    # Sauvegarder l'image résultante
    cv2.imwrite(new_image_path, closed)
    #print("L'image avec la plus grande figure blanche sur fond noir a été sauvegardée avec succès.")
    #closed = cv2.resize(closed, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return closed

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

# Trier les fichiers en ordre numérique décroissant
sorted_files = sorted(
    os.listdir(input_folder),
    key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else -1,
    reverse=True
)

for filename in sorted_files:
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

# Graphe 3D

images_input_3D = 'Images_recon'

# Liste pour stocker les images binaires
images = []

# Trier les fichiers dans le dossier images_input par ordre numérique décroissant
sorted_files = sorted(
    os.listdir(images_input_3D),
    key=lambda x: int(''.join(filter(str.isdigit, x))),  # Extraire le nombre pour le tri
    reverse=True  # Ordre décroissant
)

# Parcourir le dossier et charger les images binaires
for filename in sorted_files:
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(images_input_3D, filename)
        # Lire l'image en niveaux de gris et la convertir en binaire
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('graycsale image',img)
        if img is not None:
            # Assumer que l'image contient uniquement 0 et 1 ou 0 et 255
            img_binaire = (img > 0).astype(int)  # Convertir les valeurs 255 en 1 si nécessaire
            images.append(img_binaire)


# Création de la figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')  # Fond blanc pour un meilleur contraste

# Boucle pour placer chaque pixel de chaque image dans le graphe 3D
num_images = len(images)
cmap = cm.viridis  # Choisir une carte de couleurs pour la profondeur

for i, img in enumerate(images):
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = np.full_like(x, i)  # Chaque image correspond à une valeur z (profondeur)

    # Créer un gradient de couleurs basé sur la profondeur
    colors = cmap(i / num_images)

    # Masque pour la matière
    mask = img == 1  # Les pixels représentant la matière

    # Sous-échantillonnage : garder 1 pixel sur 5000
    indices = np.flatnonzero(mask)  # Indices linéaires des pixels valides
    selected_indices = np.random.choice(indices, size=len(indices) // 100, replace=False)  # Sous-échantillonner de 1/100 image
    final_mask = np.zeros_like(mask, dtype=bool)
    final_mask.flat[selected_indices] = True  # Construire un masque final

    # Afficher uniquement les pixels sous-échantillonnés
    ax.scatter(x[final_mask], y[final_mask], z[final_mask], c=[colors], marker='o', s=1, alpha=0.6)



# Paramètres de visualisation
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstruction 3D de l\'objet avec visualisation améliorée')

plt.show()

import os
import numpy as np
import cv2
from skimage import measure
from stl import mesh

def load_images_as_volume(input_folder):
    """
    Charge les images PNG binaires dans un volume 3D.
    
    Args:
        input_folder (str): Chemin vers le dossier contenant les images PNG.

    Returns:
        numpy.ndarray: Volume 3D (H, W, D) avec les valeurs 0 (fond) et 1 (objet).
    """
    # Liste des fichiers dans le dossier, triée par ordre numérique
    sorted_files = sorted(
    os.listdir(input_folder),
    key=lambda x: int(''.join(filter(str.isdigit, x))),  # Extraire le nombre pour le tri
    reverse=True # Ordre décroissant
    )

    # Charger chaque image et l'ajouter à la pile
    volume = []
    for filename in sorted_files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Normaliser l'image pour qu'elle soit binaire (0 ou 1)
            binary_img = (img > 127).astype(np.uint8)
            volume.append(binary_img)

    # Convertir en un tableau 3D (H, W, D)
    return np.stack(volume, axis=-1)

def create_3d_model_from_volume(volume, voxel_size=1.0, output_file="model.stl"):
    """
    Convertit un volume 3D en un fichier STL en utilisant Marching Cubes.
    
    Args:
        volume (numpy.ndarray): Tableau 3D (H, W, D) binaire avec l'objet en 1.
        voxel_size (float): Taille d'un voxel (optionnel, par défaut 1.0).
        output_file (str): Nom du fichier STL de sortie.
    """
    # Utiliser Marching Cubes pour extraire la surface
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0, spacing=(voxel_size, voxel_size, voxel_size))

    # Créer une structure pour le fichier STL
    object_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            object_mesh.vectors[i][j] = verts[face[j], :]

    # Sauvegarder le modèle STL
    object_mesh.save(output_file)
    print(f"Modèle 3D sauvegardé dans {output_file}")

# Chemin vers le dossier contenant les images binaires PNG
input_folder = images_input_3D
output_file = "model_3D_final.stl"  # Nom du fichier STL

# Charger les images et créer le volume 3D
volume = load_images_as_volume(input_folder)

# Générer le fichier STL à partir du volume
create_3d_model_from_volume(volume, voxel_size=1, output_file=output_file)

