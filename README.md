## Entraînement du modèle

Avant de lancer l'application, vous devez entraîner le modèle :

```bash
python train_model.py --data_path data/heart.csv --explore --smote
```

## Lancement de l'application

Pour lancer l'application en mode développement :

```bash
# Sur Windows
set FLASK_ENV=development
flask run

# Sur macOS/Linux
export FLASK_ENV=development
flask run
```

L'application sera accessible à l'adresse [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

## Déploiement en production

Pour le déploiement en production, plusieurs options sont disponibles :

### Option 1: Serveur WSGI (Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Option 2: Docker

Un Dockerfile est fourni pour faciliter le déploiement avec Docker :

```bash
docker build -t cardiopredict .
docker run -p 8000:8000 cardiopredict
```
