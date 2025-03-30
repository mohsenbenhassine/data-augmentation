import numpy as np
import matplotlib.pyplot as plt

def star_vertices(R=1.0, r=0.5, n_points=5):
    """
    Calcule les sommets d'une étoile à n_points (nombre de pointes).

    - R : rayon pour les sommets extérieurs.
    - r : rayon pour les sommets intérieurs.
    - n_points : nombre de pointes de l'étoile.

    Renvoie un tableau NumPy de forme (2*n_points, 2) contenant les coordonnées (x, y)
    des sommets, en alternant points extérieurs et intérieurs.
    """
    vertices = []
    angle_offset = np.pi/2  # pour que la pointe supérieure soit dirigée vers le haut
    for i in range(2 * n_points):
        angle = angle_offset + i * np.pi / n_points
        if i % 2 == 0:  # sommet extérieur
            vertices.append((R * np.cos(angle), R * np.sin(angle)))
        else:           # sommet intérieur
            vertices.append((r * np.cos(angle), r * np.sin(angle)))
    return np.array(vertices)

def generate_star_data(n_points_total=1000, R=1.0, r=0.5, n_points_star=5, noise=0.05):
    """
    Génère un jeu de données en forme d'étoile en échantillonnant des points sur
    les segments reliant les sommets d'une étoile.

    - n_points_total : nombre total de points à générer.
    - R : rayon extérieur de l'étoile.
    - r : rayon intérieur de l'étoile.
    - n_points_star : nombre de pointes de l'étoile.
    - noise : intensité du bruit gaussien ajouté aux points.

    Renvoie un tableau NumPy de forme (n_points_total, 2) contenant les coordonnées (x, y).
    """
    # Calcul des sommets de l'étoile
    vertices = star_vertices(R, r, n_points_star)
    n_segments = len(vertices)  # nombre de segments (boucle fermée)
    points_per_segment = n_points_total // n_segments  # points par segment

    data = []
    # Pour chaque segment, échantillonner linéairement entre deux sommets successifs
    for i in range(n_segments):
        start = vertices[i]
        end = vertices[(i + 1) % n_segments]  # boucle fermée
        # Créer des paramètres t variant de 0 à 1
        t = np.linspace(0, 1, points_per_segment)
        # Calculer les points le long du segment
        segment = np.outer(1 - t, start) + np.outer(t, end)
        data.append(segment)

    data = np.concatenate(data, axis=0)
    # Ajouter un peu de bruit pour que les points ne soient pas parfaitement alignés
    data += noise * np.random.randn(*data.shape)
    return data

# Générer le jeu de données en forme d'étoile
data = generate_star_data(n_points_total=100, R=1.0, r=0.5, n_points_star=5, noise=0.05)

# Visualiser le jeu de données
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], s=10, color='blue', alpha=0.6)
plt.title("Jeu de données en forme d'étoile")
plt.axis('equal')
plt.show()
