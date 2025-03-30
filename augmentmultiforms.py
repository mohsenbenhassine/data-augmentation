import matplotlib.pyplot as plt
from google.colab import files

# Définir la figure avec une taille et une résolution élevées
plt.figure(figsize=(8, 8), dpi=600)  # 600 DPI pour une netteté maximale

# Plot des données originales avec des étoiles, bordures épaisses et points gros
plt.scatter(data[:, 0], data[:, 1],
            s=100,                 # Taille des points augmentée pour plus de visibilité
            marker='*',            # Marqueur en étoile
            color='darkblue',      # Couleur bleu foncé
            alpha=0.8,             # Opacité augmentée
            edgecolors='black',    # Bordures noires
            linewidth=1.5,         # Largeur des bordures augmentée pour des bords gras
            label='Original data') # Étiquette pour la légende

# Plot des données augmentées avec des cercles, bordures épaisses et points gros
plt.scatter(generated_data[:, 0], generated_data[:, 1],
            s=50,                  # Taille des points augmentée (plus petite que les étoiles)
            marker='o',            # Marqueur en cercle
            color='red',           # Couleur rouge
            alpha=0.6,             # Transparence légèrement différente
            edgecolors='black',    # Bordures noires
            linewidth=1.5,         # Largeur des bordures augmentée pour des bords gras
            label='Aug. data')     # Étiquette pour la légende

# Titre avec texte en gras
plt.title("Original Data vs. Augmented Data",
          fontsize=16,
          fontweight='bold',
          pad=20)

# Étiquettes des axes avec texte en gras
plt.xlabel("X Axis", fontsize=14, fontweight='bold')
plt.ylabel("Y Axis", fontsize=14, fontweight='bold')

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

# Ajuster la disposition
plt.tight_layout()

# Enregistrer l'image avec une haute résolution
plt.savefig("graphique.png", dpi=600, bbox_inches="tight", format="png")

# Télécharger automatiquement le fichier
files.download("graphique.png")

# Afficher le graphique
plt.show()