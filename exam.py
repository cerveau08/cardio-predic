# Modèle de Machine Learning pour la prédiction des maladies cardiaques
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Chargement des données
print("1. Chargement des données")
df = pd.read_csv('heart.csv')
print("Dimensions du dataset:", df.shape)
print("\nAperçu des données:")
print(df.head())

# 2. Liste des différents modèles à utiliser
print("\n2. Liste des modèles pour résoudre ce problème de classification binaire:")
print("- Régression Logistique")
print("- Arbre de Décision")
print("- Random Forest")
print("- Gradient Boosting")
print("- Support Vector Machine (SVM)")
print("- K-Nearest Neighbors (KNN)")
print("- Naive Bayes")
print("- Réseau de Neurones (MLP)")

# 3. Préparation des données
print("\n3. Préparation des données")

# Informations sur les données
print("\nInformations sur les types de données:")
print(df.info())

# Statistiques descriptives
print("\nStatistiques descriptives:")
print(df.describe())

# Vérification des valeurs manquantes
print("\nVérification des valeurs manquantes:")
missing_values = df.isnull().sum()
print(missing_values)

# Si des valeurs manquantes sont présentes, les imputer
if missing_values.sum() > 0:
    print("Imputation des valeurs manquantes...")
    imputer = SimpleImputer(strategy='median')
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# Vérification des doublons
print("\nVérification des doublons:")
duplicates = df.duplicated().sum()
print(f"Nombre de doublons: {duplicates}")

# Suppression des doublons si présents
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Doublons supprimés. Nouvelles dimensions: {df.shape}")

# Vérification des valeurs aberrantes avec des boxplots
print("\nAnalyse des valeurs aberrantes (résumé statistique):")
plt.figure(figsize=(15, 10))
df.boxplot(figsize=(15, 10))
plt.title('Boxplots pour détecter les valeurs aberrantes')
plt.savefig('boxplots.png')
plt.close()
print("Boxplots sauvegardés dans 'boxplots.png'")

# Traitement des valeurs aberrantes si nécessaire
# On applique une méthode d'écrêtage (capping) pour les valeurs numériques
for column in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    if outliers_count > 0:
        print(f"Colonne {column}: {outliers_count} valeurs aberrantes détectées et traitées")
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

# Vérification de l'équilibre des classes
print("\nVérification de l'équilibre des classes:")
class_distribution = df['target'].value_counts(normalize=True) * 100
print(class_distribution)

# Visualisation de la distribution des classes
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution de la variable cible')
plt.savefig('class_distribution.png')
plt.close()
print("Distribution des classes sauvegardée dans 'class_distribution.png'")

# Appliquer SMOTE si déséquilibre important (seuil arbitraire de 60/40)
imbalance_threshold = 0.4
minority_class_ratio = min(class_distribution) / 100
if minority_class_ratio < imbalance_threshold:
    print(f"Déséquilibre de classe détecté: la classe minoritaire représente {minority_class_ratio:.2f} du dataset")
    print("Application de SMOTE pour rééquilibrer les données...")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Remplacer les données originales
    df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                    pd.DataFrame(y_resampled, columns=['target'])], axis=1)
    
    print(f"Nouvelles dimensions après SMOTE: {df.shape}")
    print("Nouvelle distribution des classes:")
    print(df['target'].value_counts(normalize=True) * 100)
    
    # Visualisation après SMOTE
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribution de la variable cible après SMOTE')
    plt.savefig('class_distribution_after_smote.png')
    plt.close()
    print("Nouvelle distribution sauvegardée dans 'class_distribution_after_smote.png'")

# Normalisation des données
print("\nNormalisation des données:")
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Statistiques après normalisation:")
print(X_scaled_df.describe().T[['mean', 'std', 'min', 'max']])

# 4. Division des données en ensembles d'entraînement et de test
print("\n4. Division des données: 70% entraînement, 30% test")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"Dimensions des données d'entraînement: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Dimensions des données de test: X_test {X_test.shape}, y_test {y_test.shape}")

# 5. Création et entraînement des modèles
print("\n5. Création et entraînement des modèles")

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

# Dictionnaires pour stocker les résultats
trained_models = {}
results = {}

# Entraînement des modèles
for name, model in models.items():
    print(f"Entraînement du modèle: {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Stockage des résultats
    results[name] = {
        'y_pred': y_pred,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'conf_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"Entraînement terminé pour {name}")

# 6. Évaluation des modèles - Matrices de confusion
print("\n6. Évaluation des modèles - Matrices de confusion")

# Fonction pour tracer la matrice de confusion
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig(filename)
    plt.close()

# Tracer et analyser les matrices de confusion
for name, result in results.items():
    cm = result['conf_matrix']
    cm_filename = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plot_confusion_matrix(cm, f"Matrice de confusion - {name}", cm_filename)
    
    # Analyser la matrice de confusion
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print(f"\nAnalyse de la matrice de confusion pour {name}:")
    print(f"Vrais Positifs (TP): {tp} ({tp/total:.2%})")
    print(f"Vrais Négatifs (TN): {tn} ({tn/total:.2%})")
    print(f"Faux Positifs (FP): {fp} ({fp/total:.2%})")
    print(f"Faux Négatifs (FN): {fn} ({fn/total:.2%})")
    print(f"Matrice de confusion sauvegardée dans '{cm_filename}'")

# 7. Calcul des métriques pour chaque modèle
print("\n7. Métriques d'évaluation pour chaque modèle")

# Tableau des métriques
metrics_df = pd.DataFrame({
    'Modèle': list(results.keys()),
    'Accuracy': [result['accuracy'] for result in results.values()],
    'Precision': [result['precision'] for result in results.values()],
    'Recall': [result['recall'] for result in results.values()]
})

print(metrics_df.to_string(index=False))

# Visualisation des métriques
plt.figure(figsize=(12, 8))
metrics_df.set_index('Modèle')[['Accuracy', 'Precision', 'Recall']].plot(kind='bar')
plt.title('Comparaison des métriques par modèle')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.close()
print("Comparaison des métriques sauvegardée dans 'metrics_comparison.png'")

# 8. Comparaison des modèles
print("\n8. Étude comparative des modèles")

# Calculer un score global (moyenne pondérée de accuracy, precision et recall)
metrics_df['Score_Global'] = (metrics_df['Accuracy'] + metrics_df['Precision'] + metrics_df['Recall']) / 3
metrics_df = metrics_df.sort_values(by='Score_Global', ascending=False)

print("Classement des modèles selon le score global:")
print(metrics_df.to_string(index=False))

# Identifier le meilleur modèle
best_model_name = metrics_df.iloc[0]['Modèle']
best_model = trained_models[best_model_name]
best_score = metrics_df.iloc[0]['Score_Global']

print(f"\nLe meilleur modèle est: {best_model_name} avec un score global de {best_score:.4f}")
print("\nDétail des performances du meilleur modèle:")
print(f"Accuracy: {metrics_df.iloc[0]['Accuracy']:.4f}")
print(f"Precision: {metrics_df.iloc[0]['Precision']:.4f}")
print(f"Recall: {metrics_df.iloc[0]['Recall']:.4f}")

# Rapport de classification détaillé pour le meilleur modèle
y_pred_best = results[best_model_name]['y_pred']
print("\nRapport de classification pour le meilleur modèle:")
print(classification_report(y_test, y_pred_best))

# 9. Sauvegarde du meilleur modèle
print("\n9. Sauvegarde du meilleur modèle")
model_filename = f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
joblib.dump(best_model, model_filename)
scaler_filename = "scaler.pkl"
joblib.dump(scaler, scaler_filename)

print(f"Meilleur modèle sauvegardé dans '{model_filename}'")
print(f"Scaler sauvegardé dans '{scaler_filename}'")
