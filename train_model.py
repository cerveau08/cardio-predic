"""
Script d'entraînement du modèle de prédiction des maladies cardiaques
Ce script charge les données, effectue le prétraitement, entraîne plusieurs modèles,
évalue leurs performances et sauvegarde le meilleur modèle.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignorer les avertissements
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Charge les données à partir d'un fichier CSV
    
    Args:
        filepath (str): Chemin vers le fichier CSV
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données
    """
    logger.info(f"Chargement des données depuis {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Données chargées avec succès. Dimensions: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        raise


def explore_data(df):
    """
    Explore les données et affiche des statistiques descriptives
    
    Args:
        df (pandas.DataFrame): DataFrame à explorer
        
    Returns:
        None
    """
    logger.info("Exploration des données")
    
    # Informations générales
    logger.info(f"Dimensions du dataset: {df.shape}")
    logger.info(f"Types de données:\n{df.dtypes}")
    
    # Statistiques descriptives
    desc_stats = df.describe()
    logger.info(f"Statistiques descriptives:\n{desc_stats}")
    
    # Valeurs manquantes
    missing_values = df.isnull().sum()
    logger.info(f"Valeurs manquantes par colonne:\n{missing_values}")
    
    # Distribution de la variable cible
    target_dist = df['target'].value_counts(normalize=True) * 100
    logger.info(f"Distribution de la variable cible:\n{target_dist}")
    
    # Visualisations
    try:
        # Créer un dossier pour les visualisations
        os.makedirs('visualizations', exist_ok=True)
        
        # Distribution de la variable cible
        plt.figure(figsize=(8, 6))
        sns.countplot(x='target', data=df)
        plt.title('Distribution de la variable cible')
        plt.savefig('visualizations/target_distribution.png')
        plt.close()
        
        # Matrice de corrélation
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
        plt.title('Matrice de corrélation des variables')
        plt.savefig('visualizations/correlation_matrix.png')
        plt.close()
        
        logger.info("Visualisations sauvegardées dans le dossier 'visualizations'")
    except Exception as e:
        logger.warning(f"Erreur lors de la création des visualisations: {str(e)}")


def preprocess_data(df, apply_smote=True):
    """
    Prétraite les données pour l'entraînement
    
    Args:
        df (pandas.DataFrame): DataFrame à prétraiter
        apply_smote (bool): Appliquer SMOTE pour rééquilibrer les classes
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler) - Données d'entraînement et de test prétraitées
    """
    logger.info("Prétraitement des données")
    
    # Vérification des valeurs manquantes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"Imputation des {missing_values.sum()} valeurs manquantes")
        imputer = SimpleImputer(strategy='median')
        df_numeric = df.select_dtypes(include=[np.number])
        df[df_numeric.columns] = imputer.fit_transform(df_numeric)
    
    # Vérification des doublons
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Suppression de {duplicates} doublons")
        df = df.drop_duplicates()
    
    # Traitement des valeurs aberrantes avec écrêtage (capping)
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        if outliers_count > 0:
            logger.info(f"Traitement de {outliers_count} valeurs aberrantes dans {column}")
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    # Séparation des features et de la variable cible
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logger.info(f"Division des données: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # Vérification de l'équilibre des classes
    class_distribution = y_train.value_counts(normalize=True) * 100
    minority_class_ratio = min(class_distribution) / 100
    
    # Appliquer SMOTE si demandé et si déséquilibre important
    if apply_smote and minority_class_ratio < 0.4:
        logger.info(f"Application de SMOTE pour rééquilibrer les classes (ratio classe minoritaire: {minority_class_ratio:.2f})")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Nouvelles dimensions après SMOTE: X_train {X_train_resampled.shape}")
        
        # Vérification de la nouvelle distribution
        new_distribution = pd.Series(y_train_resampled).value_counts(normalize=True) * 100
        logger.info(f"Nouvelle distribution des classes: {new_distribution.to_dict()}")
        
        X_train = X_train_resampled
        y_train = y_train_resampled
    
    # Normalisation des données
    logger.info("Normalisation des données avec StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train, grid_search=False):
    """
    Entraîne plusieurs modèles de classification
    
    Args:
        X_train (numpy.ndarray): Features d'entraînement
        y_train (numpy.ndarray): Cibles d'entraînement
        grid_search (bool): Utiliser GridSearchCV pour l'optimisation des hyperparamètres
        
    Returns:
        dict: Dictionnaire des modèles entraînés
    """
    logger.info("Entraînement des modèles")
    
    # Initialisation des modèles
    models = {
        'Régression Logistique': LogisticRegression(random_state=42, max_iter=1000),
        'Arbre de Décision': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    # Paramètres pour GridSearchCV
    if grid_search:
        param_grids = {
            'Régression Logistique': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            },
            'Arbre de Décision': {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    # Dictionnaire pour stocker les modèles entraînés
    trained_models = {}
    
    # Entraînement des modèles
    for name, model in models.items():
        logger.info(f"Entraînement du modèle: {name}")
        
        try:
            if grid_search and name in param_grids:
                logger.info(f"Optimisation des hyperparamètres avec GridSearchCV pour {name}")
                grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train, y_train)
                
                best_model = grid.best_estimator_
                logger.info(f"Meilleurs paramètres pour {name}: {grid.best_params_}")
                logger.info(f"Meilleur score CV pour {name}: {grid.best_score_:.4f}")
                
                trained_models[name] = best_model
            else:
                model.fit(X_train, y_train)
                trained_models[name] = model
            
            logger.info(f"Entraînement terminé pour {name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de {name}: {str(e)}")
    
    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Évalue les performances des modèles
    
    Args:
        models (dict): Dictionnaire des modèles entraînés
        X_test (numpy.ndarray): Features de test
        y_test (numpy.ndarray): Cibles de test
        
    Returns:
        pandas.DataFrame: DataFrame des métriques d'évaluation
    """
    logger.info("Évaluation des modèles")
    
    # Dictionnaire pour stocker les résultats
    results = {}
    
    # Évaluer chaque modèle
    for name, model in models.items():
        logger.info(f"Évaluation du modèle: {name}")
        
        try:
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Calcul des métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            # Stocker les résultats
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred
            }
            
            # Afficher les résultats
            logger.info(f"Résultats pour {name}:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-score: {f1:.4f}")
            
            # Afficher la matrice de confusion
            logger.info(f"  Matrice de confusion:\n{cm}")
            
            # Créer et sauvegarder la visualisation de la matrice de confusion
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Matrice de confusion - {name}')
            plt.ylabel('Vraie classe')
            plt.xlabel('Classe prédite')
            plt.savefig(f'visualizations/confusion_matrix_{name.replace(" ", "_").lower()}.png')
            plt.close()
            
            # Rapport de classification détaillé
            report = classification_report(y_test, y_pred)
            logger.info(f"  Rapport de classification:\n{report}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de {name}: {str(e)}")
    
    # Créer un DataFrame des métriques pour comparaison
    metrics_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'Accuracy': [result['accuracy'] for result in results.values()],
        'Precision': [result['precision'] for result in results.values()],
        'Recall': [result['recall'] for result in results.values()],
        'F1-score': [result['f1_score'] for result in results.values()]
    })
    
    # Calculer un score global (moyenne des métriques)
    metrics_df['Score_Global'] = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-score']].mean(axis=1)
    
    # Trier par score global
    metrics_df = metrics_df.sort_values(by='Score_Global', ascending=False).reset_index(drop=True)
    
    # Afficher le tableau des métriques
    logger.info(f"Comparaison des métriques:\n{metrics_df.to_string(index=False)}")
    
    # Visualisation comparative des métriques
    plt.figure(figsize=(12, 8))
    metrics_df.set_index('Modèle')[['Accuracy', 'Precision', 'Recall', 'F1-score']].plot(kind='bar')
    plt.title('Comparaison des métriques par modèle')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/metrics_comparison.png')
    plt.close()
    
    return metrics_df, results


def save_best_model(models, metrics_df, scaler, output_dir='models'):
    """
    Sauvegarde le meilleur modèle et le scaler
    
    Args:
        models (dict): Dictionnaire des modèles entraînés
        metrics_df (pandas.DataFrame): DataFrame des métriques d'évaluation
        scaler (sklearn.preprocessing.StandardScaler): Scaler entraîné
        output_dir (str): Dossier de sortie pour sauvegarder les modèles
        
    Returns:
        tuple: (best_model_name, best_model_path) - Nom et chemin du meilleur modèle
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Identifier le meilleur modèle
    best_model_name = metrics_df.iloc[0]['Modèle']
    best_model = models[best_model_name]
    best_score = metrics_df.iloc[0]['Score_Global']
    
    logger.info(f"Meilleur modèle: {best_model_name} avec un score global de {best_score:.4f}")
    
    # Chemin pour sauvegarder le meilleur modèle
    best_model_filename = f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
    best_model_path = os.path.join(output_dir, best_model_filename)
    
    # Sauvegarder le meilleur modèle
    logger.info(f"Sauvegarde du meilleur modèle dans {best_model_path}")
    joblib.dump(best_model, best_model_path)
    
    # Sauvegarder également le scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    logger.info(f"Sauvegarde du scaler dans {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    # Sauvegarder le tableau des métriques au format CSV
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Métriques sauvegardées dans {metrics_path}")
    
    return best_model_name, best_model_path


def main(args):
    """
    Fonction principale pour l'entraînement du modèle
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        None
    """
    # Charger les données
    df = load_data(args.data_path)
    
    # Explorer les données
    if args.explore:
        explore_data(df)
    
    # Prétraiter les données
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, apply_smote=args.smote)
    
    # Entraîner les modèles
    trained_models = train_models(X_train, y_train, grid_search=args.grid_search)
    
    # Évaluer les modèles
    metrics_df, results = evaluate_models(trained_models, X_test, y_test)
    
    # Sauvegarder le meilleur modèle
    best_model_name, best_model_path = save_best_model(trained_models, metrics_df, scaler, args.output_dir)
    
    logger.info("Entraînement terminé avec succès!")
    logger.info(f"Meilleur modèle: {best_model_name}")
    logger.info(f"Sauvegardé dans: {best_model_path}")


if __name__ == "__main__":
    # Analyser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Script d'entraînement pour le modèle de prédiction des maladies cardiaques")
    
    parser.add_argument('--data_path', type=str, default='data/heart.csv', 
                        help='Chemin vers le fichier de données CSV')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Dossier de sortie pour sauvegarder les modèles')
    parser.add_argument('--explore', action='store_true',
                        help='Explorer les données et générer des visualisations')
    parser.add_argument('--smote', action='store_true',
                        help='Appliquer SMOTE pour rééquilibrer les classes')
    parser.add_argument('--grid_search', action='store_true',
                        help='Utiliser GridSearchCV pour l\'optimisation des hyperparamètres')
    
    args = parser.parse_args()
    
    # Exécuter la fonction principale
    main(args)