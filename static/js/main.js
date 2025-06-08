/**
 * Script principal pour l'application CardioPredict
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialisation de tous les tooltips Bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialisation de tous les popovers Bootstrap
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Gestion des messages flash auto-fermants
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        // Auto-fermer après 5 secondes
        setTimeout(function() {
            const closeButton = alert.querySelector('.btn-close');
            if (closeButton) {
                closeButton.click();
            }
        }, 5000);
    });

    // Animation de défilement fluide pour les ancres
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70, // offset pour la navbar fixe
                    behavior: 'smooth'
                });
            }
        });
    });

    // Animation lors du défilement
    function animateOnScroll() {
        const elements = document.querySelectorAll('.animate-on-scroll');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            if (elementPosition < windowHeight - 50) {
                const animationClass = element.dataset.animation || 'fade-in';
                element.classList.add(animationClass);
            }
        });
    }
    
    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Exécuter une fois au chargement de la page

    // Fonctionnalité de retour en haut
    const backToTopButton = document.getElementById('back-to-top');
    if (backToTopButton) {
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                backToTopButton.classList.add('show');
            } else {
                backToTopButton.classList.remove('show');
            }
        });

        backToTopButton.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // Gestion du mode sombre
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        // Vérifier la préférence de mode sombre enregistrée
        const isDarkMode = localStorage.getItem('darkMode') === 'enabled';
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
            darkModeToggle.checked = true;
        }

        darkModeToggle.addEventListener('change', function() {
            if (this.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'enabled');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('darkMode', 'disabled');
            }
        });
    }

    // Animation des icônes de statistiques dans le tableau de bord
    const statIcons = document.querySelectorAll('.stat-card .display-4 i');
    if (statIcons.length > 0) {
        statIcons.forEach((icon, index) => {
            setTimeout(() => {
                icon.classList.add('animate__animated', 'animate__bounceIn');
            }, index * 200);
        });
    }

    // Mise en évidence des lignes de tableau au survol
    const tableRows = document.querySelectorAll('table tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.classList.add('table-hover-highlight');
        });
        row.addEventListener('mouseleave', function() {
            this.classList.remove('table-hover-highlight');
        });
    });

    // Animation des barres de progression
    function animateProgressBars() {
        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            const value = bar.getAttribute('aria-valuenow');
            bar.style.width = '0%';
            
            setTimeout(() => {
                bar.style.transition = 'width 1s ease-in-out';
                bar.style.width = value + '%';
            }, 200);
        });
    }
    
    animateProgressBars();

    // Gestion des onglets dans l'application
    const tabLinks = document.querySelectorAll('.nav-tabs .nav-link');
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Supprimer la classe active de tous les onglets
            tabLinks.forEach(tab => tab.classList.remove('active'));
            
            // Ajouter la classe active à l'onglet cliqué
            this.classList.add('active');
            
            // Cacher tous les contenus d'onglets
            const tabContents = document.querySelectorAll('.tab-content .tab-pane');
            tabContents.forEach(content => content.classList.remove('show', 'active'));
            
            // Afficher le contenu de l'onglet cliqué
            const targetId = this.getAttribute('href');
            const targetContent = document.querySelector(targetId);
            if (targetContent) {
                targetContent.classList.add('show', 'active');
            }
        });
    });

    // Ajout d'interactions sur les cartes
    const interactiveCards = document.querySelectorAll('.interactive-card');
    interactiveCards.forEach(card => {
        card.addEventListener('click', function() {
            // Toggle la classe pour agrandir/réduire la carte
            this.classList.toggle('card-expanded');
            
            // Afficher/masquer le contenu détaillé
            const cardDetails = this.querySelector('.card-details');
            if (cardDetails) {
                if (cardDetails.style.maxHeight) {
                    cardDetails.style.maxHeight = null;
                } else {
                    cardDetails.style.maxHeight = cardDetails.scrollHeight + 'px';
                }
            }
        });
    });
});