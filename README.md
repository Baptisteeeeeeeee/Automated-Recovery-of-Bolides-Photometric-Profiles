# Automated-Recovery-of-Bolides-Photometric-Profiles
Dans un premier temps, les étoiles sont segmentées par une architecture de type U-Net puis identifiées automatiquement à l'aide d'Astrometry.net.
 ![011023_014527_br_with_stars](https://github.com/user-attachments/assets/e61b9374-22a8-4222-b8de-6884bf6cc6b4)

Le modèle de segmentation du bolide est ensuite appliqué à chaque frame de la vidéo. La zone de mouvement du bolide est précédemment déterminée par flot optique afin de réudire le nombre d'inférences aux modèles.
Pour chaque frame où le bolide est détecté, une visualisation de la zone de mouvement du bolide, de son masque de segmentation et du point utilisé pour calculer sa luminosité est disponible pour l'utilisateur afin d'améliorer le controle. 

![image](https://github.com/user-attachments/assets/58665c0a-85d4-4bb0-82f5-ee783269ac8a)

Grâce à ces informations, let en utilisant l'équation ci-dessous, le profil photomètrique des bolides peut être déterminé :
<img width="635" height="76" alt="Capture d’écran 2025-09-08 à 17 33 46" src="https://github.com/user-attachments/assets/ff24e944-3e78-4e05-bf54-959bd450768e" />


![photometric_profile](https://github.com/user-attachments/assets/1777c2e8-bbde-4493-9737-477aba5ec4e3)


## Création du jeu de données
Le jeu de données a été créé en éliminant les étoiles originales des images par ouverture morphologique locale avant d'en insérer de nouvelles, créée artificiellement de forme et position connues.
![star_free_image](https://github.com/user-attachments/assets/efe529ee-5bc2-48dd-884e-9cb23d384d27)
![synthetic_stars_image](https://github.com/user-attachments/assets/834cfc3c-83d1-449a-b731-b34a0c809929)


## Pré-traitements des données 

L'un des problèmes majeurs de la détection d'étoiles est le risque de faux positifs induits par les lumières artificielles dans les zones urbaines. Un modèle est donc utilisé en amont de la détection d'étoiles pour segmenter ces zones et prévenir les fausses détections.
![not_preprocessed](https://github.com/user-attachments/assets/7ba6e343-2225-47b2-ae84-f4a4ea3785d4)

<img width="852" height="569" alt="preprocessed" src="https://github.com/user-attachments/assets/ec054650-9762-4f8a-8db2-4f6047f2c38c" />

