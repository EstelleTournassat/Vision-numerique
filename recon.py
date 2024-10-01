import cv2
import glob
import numpy as np
from scipy.ndimage import rotate

# Fonction laminogram sans les .geo

def laminogram(sinogram):
    # Calculer les paramètres à partir du sinograme
    nbprj, nbpix = sinogram.shape # il faut peut être inverser nbprj et nbpix, j'ai pas testé
    
    # Initialiser une image vide
    image = np.ones((nbpix, nbpix))

    # Comme la deuxième moitié des projections est symétrique,
    # on peut traiter de seulement la première moitié
    # Le code assume un nombre impair de projections et que la première et la dernière sont au même angle
    #
    # On pourrait au lieu faire un sinograme avec une rotation de 180 degrés
    #
    for i in range(nbprj//2+1):
        # Répéter la projection sur plusieurs colonnes
        # pour l'étaler  sur toute l'image à reconstruire
        sinogram_image = np.tile(sinogram[i], (nbpix, 1))

        # Tourne la projection étalée selon l'angle à laquelle elle a été prise
        rotated_list = rotate(sinogram_image, 360*i/(nbprj-1), reshape=False, order=0)

        # Ajouter les listes tournées
        image *= rotated_list
    
    return image


images = np.array([cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob("Images/*.png")])/255
# sans boucle, test rapide
image_final = laminogram(images[:,220,:])
print(image_final)
#cv2.imshow("Title", image_final)

#print(image_final.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(0,480,10):
    
    image_final = laminogram(images[:,i,:])
    #print(image_final)
    cv2.imshow("Title", image_final)
    #cv2.waitKey(0)

    #cv2.destroyAllWindows()
    #print(images[:,i,:])
    



