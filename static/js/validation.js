/**
 * Script de validation des formulaires pour CardioPredict
 */

document.addEventListener('DOMContentLoaded', function() {
    /**
     * Validation personnalisée du formulaire de prédiction
     */
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        // Ajouter des écouteurs d'événements pour la validation en temps réel
        const numericInputs = predictionForm.querySelectorAll('input[type="number"]');
        numericInputs.forEach(input => {
            // Validation à chaque changement
            input.addEventListener('input', function() {
                validateNumericInput(this);
            });
            
            // Validation à la perte de focus
            input.addEventListener('blur', function() {
                validateNumericInput(this, true);
            });
        });
        
        // Validation des sélecteurs
        const selectInputs = predictionForm.querySelectorAll('select');
        selectInputs.forEach(select => {
            select.addEventListener('change', function() {
                validateSelectInput(this);
            });
            
            select.addEventListener('blur', function() {
                validateSelectInput(this, true);
            });
        });
        
        // Validation des boutons radio
        const radioGroups = ['sex', 'fbs', 'exang'];
        radioGroups.forEach(name => {
            const radios = predictionForm.querySelectorAll(`input[name="${name}"]`);
            radios.forEach(radio => {
                radio.addEventListener('change', function() {
                    validateRadioGroup(name);
                });
            });
        });
        
        // Validation à la soumission du formulaire
        predictionForm.addEventListener('submit', function(e) {
            let isValid = true;
            
            // Valider tous les champs numériques
            numericInputs.forEach(input => {
                if (!validateNumericInput(input, true)) {
                    isValid = false;
                }
            });
            
            // Valider tous les sélecteurs
            selectInputs.forEach(select => {
                if (!validateSelectInput(select, true)) {
                    isValid = false;
                }
            });
            
            // Valider tous les groupes de boutons radio
            radioGroups.forEach(name => {
                if (!validateRadioGroup(name, true)) {
                    isValid = false;
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                
                // Afficher un message d'erreur
                showValidationError("Veuillez corriger les erreurs dans le formulaire avant de soumettre.");
                
                // Scroll jusqu'au premier champ en erreur
                const firstInvalidField = document.querySelector('.is-invalid');
                if (firstInvalidField) {
                    firstInvalidField.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        });
    }
    
    /**
     * Validation du formulaire de prédiction par lot
     */
    const batchForm = document.getElementById('batch-form');
    if (batchForm) {
        const fileInput = batchForm.querySelector('input[type="file"]');
        
        fileInput.addEventListener('change', function() {
            validateFileInput(this);
        });
        
        batchForm.addEventListener('submit', function(e) {
            if (!validateFileInput(fileInput, true)) {
                e.preventDefault();
                showValidationError("Veuillez sélectionner un fichier CSV valide.");
            }
        });
    }
    
    /**
     * Fonctions utilitaires pour la validation
     */
    
    // Validation d'un champ numérique
    function validateNumericInput(input, showError = false) {
        const value = input.value.trim();
        const min = parseFloat(input.getAttribute('min'));
        const max = parseFloat(input.getAttribute('max'));
        const step = parseFloat(input.getAttribute('step') || 1);
        
        let isValid = true;
        let errorMessage = "";
        
        // Vérifier si le champ est vide
        if (value === "") {
            isValid = false;
            errorMessage = "Ce champ est obligatoire.";
        }
        // Vérifier si la valeur est un nombre
        else if (isNaN(parseFloat(value))) {
            isValid = false;
            errorMessage = "Veuillez entrer un nombre valide.";
        }
        // Vérifier si la valeur est dans l'intervalle autorisé
        else if (parseFloat(value) < min || parseFloat(value) > max) {
            isValid = false;
            errorMessage = `Veuillez entrer une valeur entre ${min} et ${max}.`;
        }
        // Vérifier si la valeur respecte le pas (step)
        else if (step !== 1 && Math.abs((parseFloat(value) - min) % step) > 0.0001) {
            isValid = false;
            errorMessage = `Veuillez entrer une valeur par incrément de ${step}.`;
        }
        
        // Mettre à jour l'état de validation
        updateValidationState(input, isValid, errorMessage, showError);
        
        return isValid;
    }
    
    // Validation d'un champ select
    function validateSelectInput(select, showError = false) {
        const value = select.value;
        
        let isValid = true;
        let errorMessage = "";
        
        if (value === "" || value === null) {
            isValid = false;
            errorMessage = "Veuillez sélectionner une option.";
        }
        
        updateValidationState(select, isValid, errorMessage, showError);
        
        return isValid;
    }
    
    // Validation d'un groupe de boutons radio
    function validateRadioGroup(name, showError = false) {
        const radios = document.querySelectorAll(`input[name="${name}"]`);
        let isChecked = false;
        
        radios.forEach(radio => {
            if (radio.checked) {
                isChecked = true;
            }
        });
        
        let isValid = isChecked;
        let errorMessage = "Veuillez sélectionner une option.";
        
        // Les boutons radio sont traités spécialement
        if (!isValid && showError) {
            radios.forEach(radio => {
                radio.parentElement.classList.add('text-danger');
                
                // Ajouter un message d'erreur après le dernier radio du groupe
                if (radio === radios[radios.length - 1]) {
                    let feedbackDiv = radio.closest('.form-group').querySelector('.invalid-feedback');
                    if (!feedbackDiv) {
                        feedbackDiv = document.createElement('div');
                        feedbackDiv.classList.add('invalid-feedback', 'd-block');
                        radio.closest('.form-group').appendChild(feedbackDiv);
                    }
                    feedbackDiv.textContent = errorMessage;
                }
            });
        } else {
            radios.forEach(radio => {
                radio.parentElement.classList.remove('text-danger');
                
                // Supprimer le message d'erreur
                const feedbackDiv = radio.closest('.form-group').querySelector('.invalid-feedback');
                if (feedbackDiv) {
                    feedbackDiv.remove();
                }
            });
        }
        
        return isValid;
    }
    
    // Validation d'un champ de type fichier
    function validateFileInput(input, showError = false) {
        const file = input.files[0];
        
        let isValid = true;
        let errorMessage = "";
        
        if (!file) {
            isValid = false;
            errorMessage = "Veuillez sélectionner un fichier.";
        } else {
            const fileName = file.name;
            const fileExt = fileName.split('.').pop().toLowerCase();
            
            if (fileExt !== 'csv') {
                isValid = false;
                errorMessage = "Le fichier doit être au format CSV.";
            } else if (file.size > 5 * 1024 * 1024) { // 5 Mo
                isValid = false;
                errorMessage = "La taille du fichier ne doit pas dépasser 5 Mo.";
            }
        }
        
        updateValidationState(input, isValid, errorMessage, showError);
        
        return isValid;
    }
    
    // Mettre à jour l'état de validation d'un champ
    function updateValidationState(input, isValid, errorMessage, showError) {
        // Supprimer les classes de validation précédentes
        input.classList.remove('is-valid', 'is-invalid');
        
        // Supprimer les messages d'erreur précédents
        const parent = input.parentElement;
        let feedbackDiv = parent.querySelector('.invalid-feedback');
        if (feedbackDiv) {
            feedbackDiv.remove();
        }
        
        // Appliquer la validation
        if (!isValid && showError) {
            input.classList.add('is-invalid');
            
            // Ajouter un message d'erreur
            feedbackDiv = document.createElement('div');
            feedbackDiv.classList.add('invalid-feedback');
            feedbackDiv.textContent = errorMessage;
            parent.appendChild(feedbackDiv);
        } else if (isValid && input.value !== "") {
            input.classList.add('is-valid');
        }
    }
    
    // Afficher un message d'erreur global
    function showValidationError(message) {
        // Vérifier si un message d'alerte existe déjà
        let alertDiv = document.querySelector('.validation-alert');
        
        if (!alertDiv) {
            // Créer une nouvelle alerte
            alertDiv = document.createElement('div');
            alertDiv.classList.add('alert', 'alert-danger', 'alert-dismissible', 'fade', 'show', 'validation-alert');
            alertDiv.setAttribute('role', 'alert');
            
            alertDiv.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Erreur !</strong> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Fermer"></button>
            `;
            
            // Insérer l'alerte au début du formulaire
            const form = document.querySelector('form');
            form.parentNode.insertBefore(alertDiv, form);
            
            // Auto-fermer après 5 secondes
            setTimeout(function() {
                const closeButton = alertDiv.querySelector('.btn-close');
                if (closeButton) {
                    closeButton.click();
                }
            }, 5000);
        } else {
            // Mettre à jour le message si l'alerte existe déjà
            const strongElement = alertDiv.querySelector('strong');
            if (strongElement) {
                strongElement.nextSibling.textContent = ` ${message}`;
            }
        }
    }
});