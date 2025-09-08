# Reconstruction automatisée des profils photométriques des bolides
Dans un premier temps, les étoiles sont segmentées à l’aide d’une architecture de type U-Net, puis identifiées automatiquement grâce au service Astrometry.net.
Afin de limiter les mauvaises correspondances liées aux phénomènes de distorsion, seules les étoiles dont l’erreur de position est faible sont conservées.

 ![011023_014527_br_with_stars](https://github.com/user-attachments/assets/e61b9374-22a8-4222-b8de-6884bf6cc6b4)

Le modèle de segmentation du bolide est ensuite appliqué à chaque image de la vidéo.
La zone de mouvement du bolide est au préalable déterminée par flot optique, ce qui permet de réduire le nombre d’inférences nécessaires.
Pour chaque image où le bolide est détecté, une visualisation est générée, incluant la zone de mouvement, le masque de segmentation ainsi que le point utilisé pour le calcul de la luminosité. Cette étape permet à l’utilisateur de mieux contrôler le processus.

![image](https://github.com/user-attachments/assets/58665c0a-85d4-4bb0-82f5-ee783269ac8a)

À partir de ces informations, et en utilisant l’équation ci-dessous, le profil photométrique du bolide peut être reconstruit :
<img width="635" height="76" alt="Capture d’écran 2025-09-08 à 17 33 46" src="https://github.com/user-attachments/assets/ff24e944-3e78-4e05-bf54-959bd450768e" />


![photometric_profile](https://github.com/user-attachments/assets/1777c2e8-bbde-4493-9737-477aba5ec4e3)

Rapport de projet complet : [rapport_stage.pdf](./docs/Rapport_PFE_Guivarch_Baptiste.pdf)
