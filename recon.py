def laminogram(sinogram):
    # Calculer les paramètres à partir du sinograme
    nbprj, nbpix = sinogram.shape # il faut peut être inverser nbprj et nbpix, j'ai pas testé
    
    # Initialiser une image vide
    image = np.ones((geo.nbpix, geo.nbpix))

    # Comme la deuxième moitié des projections est symétrique,
    # on peut traiter de seulement la première moitié
    # Le code assume un nombre impair de projections et que la première et la dernière sont au même angle
    #
    # On pourrait au lieu faire un sinograme avec une rotation de 180 degrés
    #
    for i in range(nbprj//2+1):
        # Répéter la projection sur plusieurs colonnes
        # pour l'étaler  sur toute l'image à reconstruire
        sinogram_image = np.tile(sinogram[i], (geo.nbpix, 1))

        # Tourne la projection étalée selon l'angle à laquelle elle a été prise
        rotated_list = rotate(sinogram_image, 360*i/(nbprj-1), reshape=False, order=0)

        # Ajouter les listes tournées
        image *= rotated_list
    
    return sinogram
