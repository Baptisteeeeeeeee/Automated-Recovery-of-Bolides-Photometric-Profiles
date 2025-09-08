# Automated-Recovery-of-Bolides-Photometric-Profiles
Dans un premier temps, les étoiles sont segmentées par une architecture de type U-Net puis identifiées automatiquement à l'aide d'Astrometry.net. Pour éviter les mauvaises identifications, possiblement dues aux phénomènes de distorsion, seule les étoiles détectées ayant des erreurs positionelles faibles sont acceptées

 ![011023_014527_br_with_stars](https://github.com/user-attachments/assets/e61b9374-22a8-4222-b8de-6884bf6cc6b4)

Le modèle de segmentation du bolide est ensuite appliqué à chaque frame de la vidéo. La zone de mouvement du bolide est précédemment déterminée par flot optique afin de réudire le nombre d'inférences aux modèles.
Pour chaque frame où le bolide est détecté, une visualisation de la zone de mouvement du bolide, de son masque de segmentation et du point utilisé pour calculer sa luminosité est disponible pour l'utilisateur afin d'améliorer le controle. 

![image](https://github.com/user-attachments/assets/58665c0a-85d4-4bb0-82f5-ee783269ac8a)

Grâce à ces informations, let en utilisant l'équation ci-dessous, le profil photomètrique des bolides peut être déterminé :
<img width="635" height="76" alt="Capture d’écran 2025-09-08 à 17 33 46" src="https://github.com/user-attachments/assets/ff24e944-3e78-4e05-bf54-959bd450768e" />


![photometric_profile](https://github.com/user-attachments/assets/1777c2e8-bbde-4493-9737-477aba5ec4e3)

Rapport de projet complett : [rapport_stage.pdf](./docs/rapport_stage.pdf)
