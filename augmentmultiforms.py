import matplotlib.pyplot as plt
from google.colab import files

# D�finir la figure avec une taille et une r�solution �lev�es
plt.figure(figsize=(8, 8), dpi=600)  # 600 DPI pour une nettet� maximale

# Plot des donn�es originales avec des �toiles, bordures �paisses et points gros
plt.scatter(data[:, 0], data[:, 1],
            s=100,                 # Taille des points augment�e pour plus de visibilit�
            marker='*',            # Marqueur en �toile
            color='darkblue',      # Couleur bleu fonc�
            alpha=0.8,             # Opacit� augment�e
            edgecolors='black',    # Bordures noires
            linewidth=1.5,         # Largeur des bordures augment�e pour des bords gras
            label='Original data') # �tiquette pour la l�gende

# Plot des donn�es augment�es avec des cercles, bordures �paisses et points gros
plt.scatter(generated_data[:, 0], generated_data[:, 1],
            s=50,                  # Taille des points augment�e (plus petite que les �toiles)
            marker='o',            # Marqueur en cercle
            color='red',           # Couleur rouge
            alpha=0.6,             # Transparence l�g�rement diff�rente
            edgecolors='black',    # Bordures noires
            linewidth=1.5,         # Largeur des bordures augment�e pour des bords gras
            label='Aug. data')     # �tiquette pour la l�gende

# Titre avec texte en gras
plt.title("Original Data vs. Augmented Data",
          fontsize=16,
          fontweight='bold',
          pad=20)

# �tiquettes des axes avec texte en gras
plt.xlabel("X Axis", fontsize=14, fontweight='bold')
plt.ylabel("Y Axis", fontsize=14, fontweight='bold')

# �tiquettes des graduations avec texte en gras
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# L�gende avec texte en gras
plt.legend(fontsize=14,
           loc='upper right',
           framealpha=0.9,
           prop={'weight': 'bold'})

# Ajouter une grille l�g�re
plt.grid(True, linestyle='--', alpha=0.7)

# Ajuster la disposition
plt.tight_layout()

# Enregistrer l'image avec une haute r�solution
plt.savefig("graphique.png", dpi=600, bbox_inches="tight", format="png")

# T�l�charger automatiquement le fichier
files.download("graphique.png")

# Afficher le graphique
plt.show()