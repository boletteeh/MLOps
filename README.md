# MLOps

Projektbeskrivelse

Dette projekt implementerer MLOps-principper til at håndtere træning, versionering og deployment af en maskinlæringsmodel. Modellen analyserer sentiment baseret på tekstdata og trænes med en dataset bestående af train_sent_emo.csv, val_sent_emo.csv og test_sent_emo.csv.

# Struktur
MLOps/

│-- .dvc/                  # DVC metadata

│-- .gitignore             # Filer, der ignoreres af Git

│-- README.md              # Projektbeskrivelse

│-- Requirements.txt       # Nødvendige Python-pakker

│-- best_model.pth.dvc     # DVC-tracked bedste model

│-- sentiment_model.pth.dvc # DVC-tracked sentiment model

│-- test.py                # Script til at evaluere modellen

│-- train.py               # Script til at træne modellen

│-- train_sent_emo.csv.dvc # DVC-tracked træningsdata

│-- val_sent_emo.csv.dvc   # DVC-tracked validationsdata

│-- test_sent_emo.csv.dvc  # DVC-tracked testdata

# Hvad er DVC?
DVC (Data Version Control) bruges til at versionere store filer såsom datasæt og modeller, som ikke skal gemmes direkte i Git. I dette projekt bruges DVC til at tracke:

Trænings-, validerings- og testdata (*.csv.dvc)

Modelfiler (*.pth.dvc)

Når DVC bruges, gemmes kun metadata i Git (*.dvc-filer), mens de faktiske filer ligger i en ekstern storage (f.eks. S3).

For at hente de nyeste versioner af disse filer:

dvc pull

For at uploade opdaterede filer:

dvc push

# Hvad er .gitignore?
.gitignore er en fil, der fortæller Git, hvilke filer der skal ignoreres. Dette er vigtigt, fordi vi ikke ønsker at versionere store filer som modeller og dataset direkte i Git.



# Installation
1. Klon projektet:
   
git clone https://github.com/boletteeh/MLOps.git

cd MLOps

2. Installer afhængigheder:
   
pip install -r Requirements.txt

3. Download data og modeller med DVC:
   
dvc pull

# Versionsstyring med DVC
Når der er nye modelopdateringer eller dataændringer, kan du tracke dem med:

dvc add best_model.pth sentiment_model.pth train_sent_emo.csv val_sent_emo.csv test_sent_emo.csv

git add *.dvc

git commit -m "Opdateret model og data"

git push origin main

dvc push
