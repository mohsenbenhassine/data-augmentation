import numpy as np
import matplotlib.pyplot as plt

def star_vertices(R=1.0, r=0.5, n_points=5):
    """
    Calcule les sommets d'une �toile � n_points (nombre de pointes).

    - R : rayon pour les sommets ext�rieurs.
    - r : rayon pour les sommets int�rieurs.
    - n_points : nombre de pointes de l'�toile.

    Renvoie un tableau NumPy de forme (2*n_points, 2) contenant les coordonn�es (x, y)
    des sommets, en alternant points ext�rieurs et int�rieurs.
    """
    vertices = []
    angle_offset = np.pi/2  # pour que la pointe sup�rieure soit dirig�e vers le haut
    for i in range(2 * n_points):
        angle = angle_offset + i * np.pi / n_points
        if i % 2 == 0:  # sommet ext�rieur
            vertices.append((R * np.cos(angle), R * np.sin(angle)))
        else:           # sommet int�rieur
            vertices.append((r * np.cos(angle), r * np.sin(angle)))
    return np.array(vertices)

def generate_star_data(n_points_total=1000, R=1.0, r=0.5, n_points_star=5, noise=0.05):
    """
    G�n�re un jeu de donn�es en forme d'�toile en �chantillonnant des points sur
    les segments reliant les sommets d'une �toile.

    - n_points_total : nombre total de points � g�n�rer.
    - R : rayon ext�rieur de l'�toile.
    - r : rayon int�rieur de l'�toile.
    - n_points_star : nombre de pointes de l'�toile.
    - noise : intensit� du bruit gaussien ajout� aux points.

    Renvoie un tableau NumPy de forme (n_points_total, 2) contenant les coordonn�es (x, y).
    """
    # Calcul des sommets de l'�toile
    vertices = star_vertices(R, r, n_points_star)
    n_segments = len(vertices)  # nombre de segments (boucle ferm�e)
    points_per_segment = n_points_total // n_segments  # points par segment

    data = []
    # Pour chaque segment, �chantillonner lin�airement entre deux sommets successifs
    for i in range(n_segments):
        start = vertices[i]
        end = vertices[(i + 1) % n_segments]  # boucle ferm�e
        # Cr�er des param�tres t variant de 0 � 1
        t = np.linspace(0, 1, points_per_segment)
        # Calculer les points le long du segment
        segment = np.outer(1 - t, start) + np.outer(t, end)
        data.append(segment)

    data = np.concatenate(data, axis=0)
    # Ajouter un peu de bruit pour que les points ne soient pas parfaitement align�s
    data += noise * np.random.randn(*data.shape)
    return data

# G�n�rer le jeu de donn�es en forme d'�toile
data = generate_star_data(n_points_total=100, R=1.0, r=0.5, n_points_star=5, noise=0.05)

# Visualiser le jeu de donn�es
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], s=10, color='blue', alpha=0.6)
plt.title("Jeu de donn�es en forme d'�toile")
plt.axis('equal')
plt.show()
