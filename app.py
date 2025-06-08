from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
import numpy as np
import pandas as pd
import joblib
import os
import json
from werkzeug.utils import secure_filename
import matplotlib
# Détection automatique de l'environnement
if os.environ.get('DYNO') or os.environ.get('PORT'):  # Heroku
    matplotlib.use('Agg')
else:  # Local
    matplotlib.use('TkAgg')  # Interface graphique pour local
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Configuration adaptative selon l'environnement
if os.environ.get('DYNO'):  # Production Heroku
    app.secret_key = os.environ.get('SECRET_KEY', 'heart_disease_prediction_prod_key')
    DEBUG_MODE = False
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
else:  # Développement local
    app.secret_key = 'heart_disease_prediction_dev_key'
    DEBUG_MODE = True
    HOST = '127.0.0.1'
    PORT = 5000

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Créer les dossiers nécessaires
for folder in ['uploads', 'models']:
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_models():
    """Charger le modèle et le scaler avec gestion d'erreurs robuste"""
    model_path = None
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    # Chercher le fichier de modèle dans le dossier models
    if os.path.exists('models'):
        pkl_files = [f for f in os.listdir('models') if f.endswith('.pkl') and f.startswith('best_model')]
        
        # Filtrer les fichiers non vides et trier par taille
        valid_files = []
        for f in pkl_files:
            file_path = os.path.join('models', f)
            if os.path.getsize(file_path) > 0:
                valid_files.append((f, os.path.getsize(file_path)))
        
        if valid_files:
            valid_files.sort(key=lambda x: x[1], reverse=True)
            model_path = os.path.join('models', valid_files[0][0])
            print(f"🔍 Fichier de modèle trouvé: {valid_files[0][0]} ({valid_files[0][1]} bytes)")
    
    try:
        if model_path and os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"📁 Chargement du modèle depuis: {model_path}")
            print(f"📁 Chargement du scaler depuis: {scaler_path}")
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            print(f"✅ Modèle et scaler chargés avec succès")
            print(f"🤖 Type de modèle: {type(model).__name__}")
            return model, scaler
        else:
            print("❌ Fichiers de modèle non trouvés")
            if DEBUG_MODE:  # En local, plus de détails
                print(f"   Modèle recherché: {model_path}")
                print(f"   Scaler recherché: {scaler_path}")
                print(f"   Contenu du dossier models: {os.listdir('models') if os.path.exists('models') else 'Dossier inexistant'}")
            return None, None
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {str(e)}")
        if DEBUG_MODE:  # Traceback complet en local
            import traceback
            traceback.print_exc()
        return None, None

# Charger les modèles au démarrage
print(f"🚀 Démarrage en mode: {'PRODUCTION (Heroku)' if not DEBUG_MODE else 'DÉVELOPPEMENT (Local)'}")
model, scaler = load_models()

# Fonction pour vérifier l'extension du fichier
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Configuration des features
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

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

def create_plot_safely(plot_function, *args, **kwargs):
    """Créer un graphique avec gestion d'erreurs adaptée à l'environnement"""
    try:
        result = plot_function(*args, **kwargs)
        
        if not DEBUG_MODE:  # Production: toujours fermer
            plt.close()
        else:  # Local: laisser ouvert si voulu
            plt.close()  # Fermer quand même pour éviter l'accumulation
            
        return result
    except Exception as e:
        print(f"Erreur génération graphique: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        plt.close()
        return None

# Routes
@app.route('/')
def index():
    env_info = {
        'environment': 'Production' if not DEBUG_MODE else 'Développement',
        'model_loaded': model is not None,
        'debug_mode': DEBUG_MODE
    }
    return render_template('index.html', env_info=env_info)

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html', 
                          feature_names=feature_names,
                          feature_descriptions=feature_descriptions,
                          feature_ranges=feature_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        flash('Aucun modèle n\'est chargé. Veuillez contacter l\'administrateur.', 'danger')
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
        
        # Créer un graphique de l'importance des variables (si supporté)
        def plot_feature_importance():
            if not hasattr(model, 'feature_importances_'):
                return None
                
            plt.figure(figsize=(10, 6))
            features_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=features_df)
            plt.title('Importance des variables')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        
        feature_importance_img = create_plot_safely(plot_feature_importance)
        
        return render_template('results.html', 
                              result=result, 
                              input_values=input_values,
                              feature_descriptions=feature_descriptions,
                              feature_importance_img=feature_importance_img)
                              
    except Exception as e:
        flash(f'Erreur lors de la prédiction: {str(e)}', 'danger')
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return redirect(url_for('predict_form'))

@app.route('/batch_predict', methods=['GET'])
def batch_predict_form():
    return render_template('batch_predict.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None or scaler is None:
        flash('Aucun modèle n\'est chargé. Veuillez contacter l\'administrateur.', 'danger')
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
                flash(f'Colonnes manquantes: {", ".join(missing_columns)}', 'danger')
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
            def plot_predictions_distribution():
                plt.figure(figsize=(10, 6))
                sns.countplot(x='prediction', data=data)
                plt.title('Distribution des prédictions')
                plt.xlabel('Prédiction (0: Maladie, 1: Pas de maladie)')
                plt.ylabel('Nombre de patients')
                plt.tight_layout()
                
                pred_dist_buf = io.BytesIO()
                plt.savefig(pred_dist_buf, format='png', dpi=100, bbox_inches='tight')
                pred_dist_buf.seek(0)
                return base64.b64encode(pred_dist_buf.read()).decode('utf-8')
            
            pred_dist_img = create_plot_safely(plot_predictions_distribution)
            
            return render_template('batch_results.html',
                                  filename=filename,
                                  result_filename=result_filename,
                                  num_samples=len(data),
                                  num_positive=sum(predictions == 0),
                                  num_negative=sum(predictions == 1),
                                  pred_dist_img=pred_dist_img)
            
        except Exception as e:
            flash(f'Erreur lors du traitement du fichier: {str(e)}', 'danger')
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return redirect(url_for('batch_predict_form'))
        finally:
            # Nettoyer le fichier uploadé
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
    else:
        flash('Type de fichier non autorisé. Veuillez télécharger un fichier CSV.', 'danger')
        return redirect(url_for('batch_predict_form'))

@app.route('/dashboard')
def dashboard():
    if model is None or scaler is None:
        flash('Aucune donnée disponible pour le tableau de bord.', 'warning')
        return redirect(url_for('index'))
    
    # Essayer de charger les données d'origine
    data_path = os.path.join('data', 'heart.csv')
    if not os.path.exists(data_path):
        flash('Données d\'entraînement non disponibles.', 'warning')
        return redirect(url_for('index'))
    
    try:
        data = pd.read_csv(data_path)
        visualizations = {}
        
        # 1. Distribution de la variable cible
        def plot_target_distribution():
            plt.figure(figsize=(8, 6))
            target_counts = data['target'].value_counts()
            plt.pie(target_counts, labels=['Maladie cardiaque', 'Pas de maladie cardiaque'], 
                    autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF'])
            plt.title('Distribution des cas de maladies cardiaques')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        
        visualizations['target_dist'] = create_plot_safely(plot_target_distribution)
        
        # 2. Distribution par âge et sexe
        def plot_age_sex_distribution():
            plt.figure(figsize=(10, 6))
            sns.histplot(data=data, x='age', hue='sex', multiple='stack', bins=15)
            plt.title('Distribution de l\'âge par sexe')
            plt.xlabel('Âge')
            plt.ylabel('Nombre de patients')
            plt.legend(['Femme', 'Homme'])
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        
        visualizations['age_sex_dist'] = create_plot_safely(plot_age_sex_distribution)
        
        # 3. Cholestérol vs Pression artérielle
        def plot_chol_bp():
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x='chol', y='trestbps', hue='target')
            plt.title('Cholestérol vs Pression artérielle')
            plt.xlabel('Cholestérol (mg/dl)')
            plt.ylabel('Pression artérielle (mm Hg)')
            plt.legend(['Maladie cardiaque', 'Pas de maladie cardiaque'])
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        
        visualizations['chol_bp'] = create_plot_safely(plot_chol_bp)
        
        # 4. Matrice de corrélation
        def plot_correlation():
            plt.figure(figsize=(12, 10))
            corr = data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
            plt.title('Matrice de corrélation des variables')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        
        visualizations['correlation'] = create_plot_safely(plot_correlation)
        
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
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
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
            'confidence': float(max(prediction_proba) * 100),
            'environment': 'production' if not DEBUG_MODE else 'development'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Endpoint de santé pour vérifier l'état de l'application"""
    return jsonify({
        'status': 'healthy',
        'environment': 'production' if not DEBUG_MODE else 'development',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'debug_mode': DEBUG_MODE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/env')
def environment_info():
    """Page d'information sur l'environnement (utile pour debug)"""
    env_vars = {
        'DYNO': os.environ.get('DYNO', 'Non défini'),
        'PORT': os.environ.get('PORT', 'Non défini'),
        'DEBUG_MODE': DEBUG_MODE,
        'HOST': HOST,
        'MATPLOTLIB_BACKEND': matplotlib.get_backend()
    }
    
    return jsonify(env_vars)

# Gestion d'erreurs
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_code=404, error_message="Page non trouvée"), 404

@app.errorhandler(500)
def internal_error(error):
    if DEBUG_MODE:
        import traceback
        traceback.print_exc()
    return render_template('error.html', error_code=500, error_message="Erreur interne du serveur"), 500

# Point d'entrée principal - FONCTIONNE EN LOCAL ET HEROKU
if __name__ == '__main__':
    print(f"🌍 Serveur démarré sur {HOST}:{PORT}")
    print(f"🔧 Mode debug: {DEBUG_MODE}")
    print(f"📊 Modèle chargé: {model is not None}")
    
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)