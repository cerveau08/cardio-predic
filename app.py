from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
import numpy as np
import pandas as pd
import joblib
import os
import json
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.secret_key = "heart_disease_prediction_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Créer le dossier uploads s'il n'existe pas
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Créer le dossier models s'il n'existe pas
if not os.path.exists('models'):
    os.makedirs('models')

# Vérifier si le modèle existe, sinon utiliser un modèle par défaut
model_path = None
scaler_path = os.path.join('models', 'scaler.pkl')

# Chercher le fichier de modèle dans le dossier models
if os.path.exists('models'):
    pkl_files = [f for f in os.listdir('models') if f.endswith('.pkl') and f.startswith('best_model')]
    
    # Filtrer les fichiers non vides et trier par taille (le plus gros en premier)
    valid_files = []
    for f in pkl_files:
        file_path = os.path.join('models', f)
        if os.path.getsize(file_path) > 0:  # Vérifier que le fichier n'est pas vide
            valid_files.append((f, os.path.getsize(file_path)))
    
    if valid_files:
        # Prendre le fichier le plus volumineux (probablement le vrai modèle)
        valid_files.sort(key=lambda x: x[1], reverse=True)
        model_path = os.path.join('models', valid_files[0][0])
        print(f"🔍 Fichier de modèle trouvé: {valid_files[0][0]} ({valid_files[0][1]} bytes)")

try:
    if model_path and os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"📁 Tentative de chargement du modèle depuis: {model_path}")
        print(f"📁 Tentative de chargement du scaler depuis: {scaler_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print(f"✅ Modèle chargé avec succès depuis: {model_path}")
        print(f"✅ Scaler chargé avec succès depuis: {scaler_path}")
    else:
        print("❌ Fichiers de modèle non trouvés")
        if model_path:
            print(f"Modèle recherché: {model_path}")
        else:
            print("Aucun fichier de modèle valide trouvé")
        print(f"Scaler recherché: {scaler_path}")
        model = None
        scaler = None
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
    import traceback
    traceback.print_exc()
    model = None
    scaler = None

# Fonction pour vérifier l'extension du fichier
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Configuration des features attendues par le modèle
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Description des features pour l'interface utilisateur
feature_descriptions = {
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

feature_ranges = {
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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html', 
                          feature_names=feature_names,
                          feature_descriptions=feature_descriptions,
                          feature_ranges=feature_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        flash('Aucun modèle n\'est chargé. Veuillez d\'abord entraîner un modèle.', 'danger')
        return redirect(url_for('index'))

    try:
        # Récupérer les données du formulaire
        input_data = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                flash(f'La valeur pour {feature} est manquante.', 'danger')
                return redirect(url_for('predict_form'))
            
            # Convertir en type approprié
            if feature == 'oldpeak':
                input_data.append(float(value))
            else:
                input_data.append(int(value))

        # Transformer les données
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Faire la prédiction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Préparer les résultats
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Absence de maladie cardiaque' if prediction == 1 else 'Présence de maladie cardiaque',
            'confidence': float(max(prediction_proba) * 100)
        }
        
        # Stocker les entrées pour l'affichage
        input_values = dict(zip(feature_names, input_data))
        
        # Créer un graphique de l'importance des variables (si supporté par le modèle)
        feature_importance_img = None
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            features_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=features_df)
            plt.title('Importance des variables')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            feature_importance_img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        return render_template('results.html', 
                              result=result, 
                              input_values=input_values,
                              feature_descriptions=feature_descriptions,
                              feature_importance_img=feature_importance_img)
                              
    except Exception as e:
        flash(f'Erreur lors de la prédiction: {str(e)}', 'danger')
        return redirect(url_for('predict_form'))

@app.route('/batch_predict', methods=['GET'])
def batch_predict_form():
    return render_template('batch_predict.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None or scaler is None:
        flash('Aucun modèle n\'est chargé. Veuillez d\'abord entraîner un modèle.', 'danger')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('Aucun fichier téléchargé', 'danger')
        return redirect(url_for('batch_predict_form'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'danger')
        return redirect(url_for('batch_predict_form'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Charger les données
            data = pd.read_csv(filepath)
            
            # Vérifier que toutes les colonnes nécessaires sont présentes
            missing_columns = [col for col in feature_names if col not in data.columns]
            if missing_columns:
                flash(f'Colonnes manquantes dans le fichier CSV: {", ".join(missing_columns)}', 'danger')
                return redirect(url_for('batch_predict_form'))
            
            # Sélectionner uniquement les colonnes nécessaires
            X = data[feature_names]
            
            # Transformer les données
            X_scaled = scaler.transform(X)
            
            # Faire les prédictions
            predictions = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)
            
            # Ajouter les prédictions au dataframe
            data['prediction'] = predictions
            data['prediction_text'] = ['Présence de maladie cardiaque' if p == 0 else 'Absence de maladie cardiaque' for p in predictions]
            data['confidence'] = [max(p) * 100 for p in probas]
            
            # Sauvegarder les résultats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"prediction_results_{timestamp}.csv"
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            data.to_csv(result_filepath, index=False)
            
            # Visualiser la distribution des prédictions
            plt.figure(figsize=(10, 6))
            sns.countplot(x='prediction', data=data)
            plt.title('Distribution des prédictions')
            plt.xlabel('Prédiction (0: Maladie, 1: Pas de maladie)')
            plt.ylabel('Nombre de patients')
            
            pred_dist_buf = io.BytesIO()
            plt.savefig(pred_dist_buf, format='png')
            pred_dist_buf.seek(0)
            pred_dist_img = base64.b64encode(pred_dist_buf.read()).decode('utf-8')
            plt.close()
            
            return render_template('batch_results.html',
                                  filename=filename,
                                  result_filename=result_filename,
                                  num_samples=len(data),
                                  num_positive=sum(predictions == 0),
                                  num_negative=sum(predictions == 1),
                                  pred_dist_img=pred_dist_img)
            
        except Exception as e:
            flash(f'Erreur lors du traitement du fichier: {str(e)}', 'danger')
            return redirect(url_for('batch_predict_form'))
    else:
        flash('Type de fichier non autorisé. Veuillez télécharger un fichier CSV.', 'danger')
        return redirect(url_for('batch_predict_form'))

@app.route('/dashboard')
def dashboard():
    # Si aucun modèle n'est chargé, rediriger vers la page d'accueil
    if model is None or scaler is None:
        flash('Aucune donnée disponible pour le tableau de bord.', 'warning')
        return redirect(url_for('index'))
    
    # Générer quelques visualisations
    visualizations = {}
    
    try:
        # Charger les données d'origine
        data = pd.read_csv(os.path.join('data', 'heart.csv'))
        
        # 1. Distribution de la variable cible
        plt.figure(figsize=(8, 6))
        target_counts = data['target'].value_counts()
        plt.pie(target_counts, labels=['Maladie cardiaque', 'Pas de maladie cardiaque'], 
                autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF'])
        plt.title('Distribution des cas de maladies cardiaques')
        
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        visualizations['target_dist'] = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close()
        
        # 2. Distribution par âge et sexe
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='age', hue='sex', multiple='stack', bins=15)
        plt.title('Distribution de l\'âge par sexe')
        plt.xlabel('Âge')
        plt.ylabel('Nombre de patients')
        plt.legend(['Femme', 'Homme'])
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        visualizations['age_sex_dist'] = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close()
        
        # 3. Cholestérol vs Pression artérielle
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='chol', y='trestbps', hue='target')
        plt.title('Cholestérol vs Pression artérielle')
        plt.xlabel('Cholestérol (mg/dl)')
        plt.ylabel('Pression artérielle (mm Hg)')
        plt.legend(['Maladie cardiaque', 'Pas de maladie cardiaque'])
        
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png')
        buf3.seek(0)
        visualizations['chol_bp'] = base64.b64encode(buf3.read()).decode('utf-8')
        plt.close()
        
        # 4. Type de douleur thoracique par diagnostic
        plt.figure(figsize=(10, 6))
        ct = pd.crosstab(data['cp'], data['target'])
        ct.plot(kind='bar', stacked=True)
        plt.title('Type de douleur thoracique par diagnostic')
        plt.xlabel('Type de douleur thoracique')
        plt.ylabel('Nombre de patients')
        plt.legend(['Maladie cardiaque', 'Pas de maladie cardiaque'])
        
        buf4 = io.BytesIO()
        plt.savefig(buf4, format='png')
        buf4.seek(0)
        visualizations['cp_target'] = base64.b64encode(buf4.read()).decode('utf-8')
        plt.close()
        
        # 5. Matrice de corrélation
        plt.figure(figsize=(12, 10))
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
        plt.title('Matrice de corrélation des variables')
        
        buf5 = io.BytesIO()
        plt.savefig(buf5, format='png')
        buf5.seek(0)
        visualizations['correlation'] = base64.b64encode(buf5.read()).decode('utf-8')
        plt.close()
        
        # Statistiques globales
        stats = {
            'total_patients': len(data),
            'heart_disease': int(sum(data['target'] == 0)),
            'no_heart_disease': int(sum(data['target'] == 1)),
            'percent_disease': float(sum(data['target'] == 0) / len(data) * 100),
            'avg_age': float(data['age'].mean()),
            'max_age': int(data['age'].max()),
            'min_age': int(data['age'].min()),
            'avg_chol': float(data['chol'].mean()),
            'men_count': int(sum(data['sex'] == 1)),
            'women_count': int(sum(data['sex'] == 0))
        }
        
        return render_template('dashboard.html', 
                               visualizations=visualizations,
                               stats=stats)
                               
    except Exception as e:
        flash(f'Erreur lors de la génération du tableau de bord: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Aucun modèle n\'est chargé'}), 500
    
    try:
        # Récupérer les données JSON
        data = request.get_json(force=True)
        
        # Vérifier que toutes les features sont présentes
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({'error': f'Caractéristiques manquantes: {missing_features}'}), 400
        
        # Préparer les données pour la prédiction
        features = [data[f] for f in feature_names]
        features_array = np.array(features).reshape(1, -1)
        
        # Normaliser les données
        features_scaled = scaler.transform(features_array)
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0].tolist()
        
        # Préparer le résultat
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Absence de maladie cardiaque' if prediction == 1 else 'Présence de maladie cardiaque',
            'confidence': float(max(prediction_proba) * 100)
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)