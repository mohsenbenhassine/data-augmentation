import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def crescent(num_points=200, R=1.0, d=0.6, center=(0, 0)):
    """
    Generates the coordinates for the outline of a crescent shape.
    """
    half_points = num_points // 2
    theta_int = np.arccos(d / 2)
    theta1 = theta_int
    theta2 = 2 * np.pi - theta_int
    theta_vals = np.linspace(theta1, theta2, half_points)
    xA = R * np.cos(theta_vals)
    yA = R * np.sin(theta_vals)
    phi_int = np.arccos(-d / 2)
    phi1 = phi_int
    phi2 = 2 * np.pi - phi_int
    phi_vals = np.linspace(phi2, phi1, half_points)
    xB = d + R * np.cos(phi_vals)
    yB = R * np.sin(phi_vals)
    x = np.concatenate([xA, xB]) + center[0]
    y = np.concatenate([yA, yB]) + center[1]
    return x, y

def create_circle(num_points=200, R=1.0, center=(0, 0)):
    """
    Generates the coordinates for a circle outline.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + R * np.cos(theta)
    y = center[1] + R * np.sin(theta)
    return x, y

def create_asterisk(num_points=200, size=1.0, center=(0, 0)):
    """
    Generates the coordinates for an improved asterisk shape.
    """
    arms = 6
    points_per_arm = num_points // arms
    xs, ys = [], []
    angles = np.linspace(0, 2 * np.pi, arms, endpoint=False)
    for angle in angles:
        t = np.linspace(-size, size, points_per_arm)
        x_seg = center[0] + t * np.cos(angle)
        y_seg = center[1] + t * np.sin(angle)
        xs.append(x_seg)
        ys.append(y_seg)
        xs.append(np.array([np.nan]))
        ys.append(np.array([np.nan]))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return x, y

def create_gaussian_noise(num_points=200, center=(0, 0), sigma=0.5):
    """
    Generates Gaussian-distributed noise points.
    """
    x = np.random.normal(loc=center[0], scale=sigma, size=num_points)
    y = np.random.normal(loc=center[1], scale=sigma, size=num_points)
    return x, y

def main():
    # Définir la figure avec une taille et une résolution élevées
    plt.figure(figsize=(8, 8), dpi=600)  # 600 DPI pour une netteté maximale

    # Générer les coordonnées de chaque forme
    x_cres, y_cres = crescent(num_points=200, R=1.0, d=0.6, center=(-1, 1))
    x_circle, y_circle = create_circle(num_points=200, R=1.0, center=(1, 1))
    x_ast, y_ast = create_asterisk(num_points=200, size=1.0, center=(-1, -1))
    x_noise, y_noise = create_gaussian_noise(num_points=200, center=(1, -1), sigma=0.5)

    # Combiner toutes les coordonnées dans X et Y
    X = np.concatenate([x_cres, x_circle, x_ast, x_noise])
    Y = np.concatenate([y_cres, y_circle, y_ast, y_noise])

    # Tracer chaque forme avec scatter pour des points gros et bordures épaisses
    plt.scatter(x_cres, y_cres,
                s=100,                 # Taille des points augmentée
                marker='o',            # Marqueur en cercle
                color='black',         # Couleur noire
                alpha=0.8,             # Opacité augmentée
                edgecolors='black',    # Bordures noires
                linewidth=1.5,         # Largeur des bordures augmentée
                label='Crescent')

    plt.scatter(x_circle, y_circle,
                s=100,                 # Taille des points augmentée
                marker='o',            # Marqueur en cercle
                color='blue',          # Couleur bleue
                alpha=0.8,             # Opacité augmentée
                edgecolors='black',    # Bordures noires
                linewidth=1.5,         # Largeur des bordures augmentée
                label='Circle')

    plt.scatter(x_ast, y_ast,
                s=100,                 # Taille des points augmentée
                marker='*',            # Marqueur en étoile pour l'astérisque
                color='red',           # Couleur rouge
                alpha=0.8,             # Opacité augmentée
                edgecolors='black',    # Bordures noires
                linewidth=1.5,         # Largeur des bordures augmentée
                label='Asterisk')

    plt.scatter(x_noise, y_noise,
                s=100,                 # Taille des points augmentée
                marker='o',            # Marqueur en cercle
                color='green',         # Couleur verte
                alpha=0.8,             # Opacité augmentée
                edgecolors='black',    # Bordures noires
                linewidth=1.5,         # Largeur des bordures augmentée
                label='Gaussian Noise')

    # Titre avec texte en gras
    plt.title("Multiple Forms with Combined Coordinates",
              fontsize=16,
              fontweight='bold',
              pad=20)

    # Étiquettes des axes avec texte en gras
    plt.xlabel("x", fontsize=14, fontweight='bold')
    plt.ylabel("y", fontsize=14, fontweight='bold')

    # Étiquettes des graduations avec texte en gras
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Légende avec texte en gras
    plt.legend(fontsize=14,
               loc='upper right',
               framealpha=0.9,
               prop={'weight': 'bold'})

    # Ajouter une grille légère
    plt.grid(True, linestyle='--', alpha=0.7)

    # Maintenir un ratio d'aspect égal et définir les limites
    plt.axis('equal')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Ajuster la disposition
    plt.tight_layout()

    # Enregistrer l'image avec une haute résolution
    plt.savefig("graphique.png", dpi=600, bbox_inches="tight", format="png")

    # Télécharger automatiquement le fichier
    files.download("graphique.png")

    # Afficher le graphique
    plt.show()

    print("Total number of points in X:", X.shape[0])
    print("Total number of points in Y:", Y.shape[0])

    return X, Y

if __name__ == "__main__":
    X, Y = main()