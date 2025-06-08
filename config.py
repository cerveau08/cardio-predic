import os
from datetime import timedelta

class Config:
    """Configuration de base pour l'application Flask"""
    # Clé secrète pour sécuriser les sessions et les formulaires
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'heart_disease_prediction_secret_key'
    
    # Configuration des dossiers
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
    DATA_FOLDER = os.path.join(BASE_DIR, 'data')
    
    # Extensions de fichiers autorisées
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Taille maximale des fichiers téléchargés (5 MB)
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    
    # Configuration des sessions
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    
    # Configuration du cache
    SEND_FILE_MAX_AGE_DEFAULT = timedelta(hours=1)
    
    # Autres paramètres
    FEATURE_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Description des features pour l'interface utilisateur
    FEATURE_DESCRIPTIONS = {
        'age': 'Âge du patient (en années)',
        'sex': 'Sexe (0 = femme, 1 = homme)',
        'cp': 'Type de douleur thoracique (0-3)',
        'trestbps': 'Pression artérielle au repos (mm Hg)',
        'chol': 'Taux de cholestérol (mg/dl)',
        'fbs': 'Taux de sucre > 120 mg/dl (0 = non, 1 = oui)',
        'restecg': 'Résultats ECG au repos (0-2)',
        'thalach': 'Fréquence cardiaque maximale',
        'exang': 'Angine pendant l\'exercice (0 = non, 1 = oui)',
        'oldpeak': 'Dépression ST induite par l\'exercice',
        'slope': 'Pente du segment ST (0-2)',
        'ca': 'Nombre de vaisseaux principaux colorés (0-4)',
        'thal': 'Résultats du test de thalium (0-3)'
    }
    
    # Plages valides pour chaque variable
    FEATURE_RANGES = {
        'age': (20, 80),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (90, 200),
        'chol': (100, 600),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (70, 220),
        'exang': (0, 1),
        'oldpeak': (0, 6.5),
        'slope': (0, 2),
        'ca': (0, 4),
        'thal': (0, 3)
    }


class DevelopmentConfig(Config):
    """Configuration pour l'environnement de développement"""
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Configuration pour l'environnement de test"""
    DEBUG = False
    TESTING = True
    # Utiliser une base de données de test séparée si nécessaire
    # Dossier de téléchargement temporaire pour les tests
    UPLOAD_FOLDER = os.path.join(Config.BASE_DIR, 'test_uploads')


class ProductionConfig(Config):
    """Configuration pour l'environnement de production"""
    DEBUG = False
    TESTING = False
    
    # En production, utiliser une vraie clé secrète
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'heart_disease_production_secret_key'
    
    # En production, SSL/HTTPS devrait être activé
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    
    # Journalisation des erreurs dans un fichier
    LOG_FILE = 'app.log'


# Configuration par défaut
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Récupère la configuration appropriée en fonction de l'environnement"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])