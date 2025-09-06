import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ClusteringClassificationAnalyzer:
    def __init__(self):
        self.ricette_df = None
        self.scaler_ricette = StandardScaler()
        self.label_encoders_ricette = {}
        self.best_models_ricette = {}
        
    def load_data(self):
        """Carica il dataset di ricette"""
        print("=== CARICAMENTO DATI ===")
        try:
            self.ricette_df = pd.read_csv('data/ricette_reali.csv')
            
            print(f"Dataset ricette caricato: {self.ricette_df.shape}")
            return True
        except Exception as e:
            print(f"Errore nel caricamento dei dati: {e}")
            return False
    
    def prepare_ricette_features(self):
        """Prepara le feature per il clustering delle ricette"""
        print("\n=== PREPARAZIONE FEATURE RICETTE ===")
        
        # Feature numeriche
        numerical_features = [
            'numero_ingredienti', 'numero_porzioni', 'tempo_preparazione_min',
            'tempo_cottura_min', 'costo_stimato_euro', 'rating_medio',
            'numero_preparazioni', 'calorie_per_porzione', 'tempo_totale_min'
        ]
        
        # Feature categoriali da encodare
        categorical_features = [
            'tipo_cucina', 'difficolta', 'tipo_piatto', 'metodo_cottura',
            'stagionalita', 'dieta_speciale', 'occasione_consumo'
        ]
        
        # Prepara feature numeriche
        X_numerical = self.ricette_df[numerical_features].fillna(0)
        X_numerical_scaled = self.scaler_ricette.fit_transform(X_numerical)
        
        # Prepara feature categoriali
        X_categorical = []
        for feature in categorical_features:
            le = LabelEncoder()
            encoded = le.fit_transform(self.ricette_df[feature].fillna('unknown'))
            self.label_encoders_ricette[feature] = le
            X_categorical.append(encoded.reshape(-1, 1))
        
        X_categorical = np.hstack(X_categorical)
        
        # Combina tutte le feature
        X_combined = np.hstack([X_numerical_scaled, X_categorical])
        
        print(f"Feature numeriche: {len(numerical_features)}")
        print(f"Feature categoriali: {len(categorical_features)}")
        print(f"Dimensione finale feature ricette: {X_combined.shape}")
        
        return X_combined, numerical_features, categorical_features
    
    def find_optimal_clusters(self, X, dataset_name, max_k=15):
        """Trova il numero ottimale di cluster usando il metodo del gomito"""
        print(f"\n=== RICERCA CLUSTER OTTIMALI PER {dataset_name.upper()} ===")
        
        # File per salvare il modello
        model_file = f'best_kmeans_{dataset_name}.pkl'
        k_file = f'optimal_k_{dataset_name}.npy'
        
        if os.path.exists(model_file) and os.path.exists(k_file):
            with open(model_file, 'rb') as f:
                best_model = pickle.load(f)
            optimal_k = np.load(k_file)
            print(f"Modello KMeans caricato per {dataset_name} con k ottimale: {optimal_k}")
            return best_model, optimal_k
        
        # Calcola inertia per diversi k
        inertias = []
        k_models = []
        K_range = range(2, min(max_k + 1, len(X) // 2))
        
        for k in K_range:
            print(f"Calcolando per k={k}...")
            best_inertia = float('inf')
            best_model_k = None
            
            # Ripete 5 volte per ogni k per trovare il migliore
            for _ in range(5):
                kmeans = KMeans(n_clusters=k, random_state=None, init='random', n_init=10)
                kmeans.fit(X)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_model_k = kmeans
            
            inertias.append(best_inertia)
            k_models.append(best_model_k)
        
        # Trova il punto del gomito
        knee = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
        optimal_k = knee.elbow if knee.elbow else K_range[len(K_range)//2]
        
        print(f"Numero ottimale di cluster per {dataset_name}: {optimal_k}")
        
        # Plot del metodo del gomito
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-')
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'Elbow at k={optimal_k}')
        plt.xlabel('Numero di cluster (k)')
        plt.ylabel('Inertia')
        plt.title(f'Metodo del gomito - {dataset_name.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'elbow_method_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Seleziona il miglior modello
        best_model = k_models[optimal_k - 2]  # -2 perché K_range inizia da 2
        
        # Salva modello e k ottimale
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
        np.save(k_file, optimal_k)
        
        return best_model, optimal_k
    
    def analyze_clusters(self, df, cluster_labels, dataset_name, numerical_features, categorical_features):
        """Analizza e interpreta i cluster formati"""
        print(f"\n=== ANALISI CLUSTER {dataset_name.upper()} ===")
        
        # Aggiungi le etichette dei cluster al dataframe
        df_analysis = df.copy()
        df_analysis['cluster'] = cluster_labels
        
        n_clusters = len(np.unique(cluster_labels))
        
        # Analisi distribuzione cluster
        cluster_counts = np.bincount(cluster_labels)
        cluster_percentages = (cluster_counts / len(cluster_labels)) * 100
        
        print(f"Distribuzione dei cluster:")
        for i in range(n_clusters):
            print(f"Cluster {i}: {cluster_counts[i]} campioni ({cluster_percentages[i]:.1f}%)")
        
        # Grafico distribuzione cluster
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        labels = [f'Cluster {i}\n({cluster_counts[i]} elementi)' for i in range(n_clusters)]
        
        wedges, texts, autotexts = plt.pie(cluster_counts, 
                                         labels=labels, 
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        
        plt.title(f'Distribuzione Cluster - {dataset_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.savefig(f'cluster_distribution_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analisi caratteristiche per cluster - Feature Numeriche
        print(f"\n--- Caratteristiche Numeriche per Cluster ---")
        cluster_stats = {}
        
        for i in range(n_clusters):
            cluster_data = df_analysis[df_analysis['cluster'] == i]
            cluster_stats[i] = {}
            
            print(f"\nCluster {i} ({len(cluster_data)} elementi):")
            
            for feature in numerical_features:
                if feature in df_analysis.columns:
                    mean_val = cluster_data[feature].mean()
                    median_val = cluster_data[feature].median()
                    cluster_stats[i][feature] = {'mean': mean_val, 'median': median_val}
                    print(f"  {feature}: Media={mean_val:.2f}, Mediana={median_val:.2f}")
        
        # Analisi caratteristiche categoriali
        print(f"\n--- Caratteristiche Categoriali Dominanti per Cluster ---")
        
        for i in range(n_clusters):
            cluster_data = df_analysis[df_analysis['cluster'] == i]
            print(f"\nCluster {i}:")
            
            for feature in categorical_features:
                if feature in df_analysis.columns:
                    most_common = cluster_data[feature].mode()
                    if len(most_common) > 0:
                        most_common_value = most_common.iloc[0]
                        percentage = (cluster_data[feature] == most_common_value).mean() * 100
                        print(f"  {feature}: '{most_common_value}' ({percentage:.1f}%)")
        
        # Interpretazione semantica dei cluster
        self.interpret_clusters(df_analysis, n_clusters, dataset_name, numerical_features, categorical_features)
        
        return df_analysis
    
    def interpret_clusters(self, df_analysis, n_clusters, dataset_name, numerical_features, categorical_features):
        """Fornisce interpretazione semantica dei cluster"""
        print(f"\n=== INTERPRETAZIONE SEMANTICA CLUSTER {dataset_name.upper()} ===")
        
        interpretations = {}
        
        for i in range(n_clusters):
            cluster_data = df_analysis[df_analysis['cluster'] == i]
            interpretation = f"Cluster {i}: "
            characteristics = []
            
            if dataset_name == "ricette":
                # Analisi specifica per ricette
                if 'difficolta' in cluster_data.columns:
                    most_common_difficulty = cluster_data['difficolta'].mode().iloc[0]
                    characteristics.append(f"Difficoltà {most_common_difficulty}")
                
                if 'tempo_totale_min' in cluster_data.columns:
                    avg_time = cluster_data['tempo_totale_min'].mean()
                    if avg_time < 30:
                        characteristics.append("Ricette veloci")
                    elif avg_time > 90:
                        characteristics.append("Ricette elaborate")
                    else:
                        characteristics.append("Ricette medie")
                
                if 'costo_stimato_euro' in cluster_data.columns:
                    avg_cost = cluster_data['costo_stimato_euro'].mean()
                    if avg_cost < 8:
                        characteristics.append("Economiche")
                    elif avg_cost > 15:
                        characteristics.append("Costose")
                    else:
                        characteristics.append("Prezzo medio")
                
                if 'tipo_piatto' in cluster_data.columns:
                    most_common_type = cluster_data['tipo_piatto'].mode().iloc[0]
                    characteristics.append(f"Principalmente {most_common_type}")
                
                if 'dieta_speciale' in cluster_data.columns:
                    most_common_diet = cluster_data['dieta_speciale'].mode().iloc[0]
                    if most_common_diet != 'normale':
                        characteristics.append(f"Dieta {most_common_diet}")
            
            # Crea interpretazione finale
                
                if 'durata_conservazione_giorni' in cluster_data.columns:
                    avg_conservation = cluster_data['durata_conservazione_giorni'].mean()
                    if avg_conservation < 7:
                        characteristics.append("Freschi")
                    elif avg_conservation > 30:
                        characteristics.append("Lunga conservazione")
                    else:
                        characteristics.append("Media conservazione")
                
                if 'origine' in cluster_data.columns:
                    most_common_origin = cluster_data['origine'].mode().iloc[0]
                    characteristics.append(f"Origine {most_common_origin}")
            
            interpretation += ", ".join(characteristics)
            interpretations[i] = interpretation
            print(interpretation)
        
        # Salva le interpretazioni
        with open(f'cluster_interpretations_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(interpretations, f)
        
        return interpretations
    
    def train_supervised_models(self, X, y, dataset_name):
        """Addestra modelli supervisionati usando i cluster come classi"""
        print(f"\n=== ADDESTRAMENTO MODELLI SUPERVISIONATI {dataset_name.upper()} ===")
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Configurazione modelli
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        best_models = {}
        best_configs = {}
        nested_cv_scores = {}
        final_results = {}
        
        # Nested Cross Validation
        for model_name, config in models_config.items():
            print(f"\n--- Processando {model_name} ---")
            
            # GridSearchCV
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Nested CV
            nested_scores = cross_val_score(
                grid_search, X_train, y_train,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            
            nested_cv_scores[model_name] = {
                'scores': nested_scores,
                'mean': nested_scores.mean(),
                'std': nested_scores.std()
            }
            
            print(f"Nested CV Accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std() * 2:.4f})")
            
            # Addestramento finale
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            best_configs[model_name] = grid_search.best_params_
            
            print(f"Migliori parametri: {grid_search.best_params_}")
            
            # Valutazione su test set
            y_pred = grid_search.best_estimator_.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            final_results[model_name] = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'nested_cv_mean': nested_cv_scores[model_name]['mean'],
                'nested_cv_std': nested_cv_scores[model_name]['std']
            }
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
            plt.ylabel('True Cluster')
            plt.xlabel('Predicted Cluster')
            plt.savefig(f'confusion_matrix_{model_name.lower()}_{dataset_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Learning Curve per classificazione
            self.plot_learning_curve_classification(
                grid_search.best_estimator_, 
                f'{model_name}_classifier', 
                X_train, y_train, 
                dataset_name
            )
        
        # Salva risultati
        with open(f'best_models_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(best_models, f)
        with open(f'best_configs_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(best_configs, f)
        with open(f'final_results_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(final_results, f)
        
        # Confronto finale
        comparison_df = pd.DataFrame(final_results).T
        comparison_df = comparison_df.round(4)
        comparison_df.to_csv(f'model_comparison_results_{dataset_name}.csv')
        
        print(f"\n=== CONFRONTO FINALE MODELLI {dataset_name.upper()} ===")
        print(comparison_df)
        
        # Plot comparativo
        self.plot_model_comparison(final_results, dataset_name)
        
        return best_models, final_results
    
    def plot_model_comparison(self, final_results, dataset_name):
        """Crea grafici comparativi dei modelli"""
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        models = list(final_results.keys())
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            values = [final_results[model][metric] for model in models]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Modelli')
        ax.set_ylabel('Score')
        ax.set_title(f'Confronto Metriche tra Modelli - {dataset_name.capitalize()}')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'model_metrics_comparison_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Esegue l'analisi completa sul dataset ricette"""
        print("=== AVVIO ANALISI CLUSTERING + CLASSIFICAZIONE ===")
        
        # Carica dati
        if not self.load_data():
            return
        
        # ANALISI RICETTE
        print("\n" + "="*60)
        print("ANALISI DATASET RICETTE")
        print("="*60)
        
        X_ricette, num_features_ricette, cat_features_ricette = self.prepare_ricette_features()
        kmeans_ricette, optimal_k_ricette = self.find_optimal_clusters(X_ricette, "ricette")
        cluster_labels_ricette = kmeans_ricette.labels_
        
        df_ricette_analyzed = self.analyze_clusters(
            self.ricette_df, cluster_labels_ricette, "ricette", 
            num_features_ricette, cat_features_ricette
        )
        
        best_models_ricette, results_ricette = self.train_supervised_models(
            X_ricette, cluster_labels_ricette, "ricette"
        )
        
        # Salva scalers e encoders
        with open('scalers_and_encoders.pkl', 'wb') as f:
            pickle.dump({
                'scaler_ricette': self.scaler_ricette,
                'label_encoders_ricette': self.label_encoders_ricette
            }, f)
        
        print("\n" + "="*60)
        print("ANALISI COMPLETATA!")
        print("="*60)
        print("File generati:")
        print("- Modelli clustering: best_kmeans_ricette.pkl")
        print("- Modelli classificazione: best_models_ricette.pkl")
        print("- Risultati: model_comparison_results_ricette.csv")
        print("- Interpretazioni cluster: cluster_interpretations_ricette.pkl")
        print("- Grafici: vari file .png per visualizzazioni")
        print("- Preprocessors: scalers_and_encoders.pkl")

    def plot_learning_curve_classification(self, estimator, title, X, y, dataset_name, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
        """Genera learning curve per modelli di classificazione"""
        plt.figure(figsize=(10, 6))
        
        # Calcola learning curve con accuracy
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        # Calcola media e std
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Stampa statistiche finali
        print(f"\n=== Learning Curve Stats - {title} ===")
        print(f"Training Accuracy: {train_scores_mean[-1]:.3f} ± {train_scores_std[-1]:.3f}")
        print(f"Validation Accuracy: {test_scores_mean[-1]:.3f} ± {test_scores_std[-1]:.3f}")
        
        # Plot
        plt.grid(True, alpha=0.3)
        plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.3, color="blue")
        plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.3, color="red")
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="blue", label="Training Set")
        plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="red", label="Validation Set")
        
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title(f'Learning Curve - {title.replace("_", " ").title()}')
        plt.legend(loc="best")
        
        # Analisi overfitting/underfitting
        final_train_acc = train_scores_mean[-1]
        final_val_acc = test_scores_mean[-1]
        gap = final_train_acc - final_val_acc
        
        if gap > 0.15:
            status = "Overfitting Detected"
            color = 'red'
        elif final_val_acc < 0.6:
            status = "Possible Underfitting"
            color = 'orange'
        else:
            status = "Good Balance"
            color = 'green'
        
        plt.text(0.05, 0.95, f"{status}\nGap: {gap:.3f}", 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
                 verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'learning_curve_{dataset_name}_{title.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve salvata: learning_curve_{dataset_name}_{title.lower()}.png")


def main():
    """Funzione principale"""
    analyzer = ClusteringClassificationAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
