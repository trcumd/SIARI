import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Caricamento del dataset
print("=== CARICAMENTO DATASET ===")
df = pd.read_csv('data/ingredienti_reali.csv', on_bad_lines='skip')
print(f"Dimensioni dataset: {df.shape}")

# Analisi preliminare del target
print("\n=== ANALISI TARGET (densita_nutrizionale_kcal100g) ===")
target_col = 'densita_nutrizionale_kcal100g'
print(f"Statistiche descrittive del target:")
print(df[target_col].describe())

# Preprocessing delle features
print("\n=== PREPROCESSING FEATURES ===")

# Separazione features numeriche e categoriche
numeric_features = ['peso_medio_g', 'durata_conservazione_giorni', 'prezzo_per_kg', 
                   'contenuto_acqua_perc', 'indice_glicemico']

categorical_features = ['categoriaAlimentare', 'colore', 'consistenza', 'origine', 
                       'stagionalita', 'provenienza_geografica', 'metodo_produzione', 
                       'livello_trasformazione']

# Verifica correlazioni tra features numeriche e target
correlations = df[numeric_features + [target_col]].corr()[target_col].sort_values(ascending=False)

# Encoding delle variabili categoriche
df_encoded = df.copy()

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Preparazione dataset per ML
feature_columns = numeric_features + [col + '_encoded' for col in categorical_features]
X = df_encoded[feature_columns]
y = df_encoded[target_col]

# Analisi features più importanti usando correlazione
feature_importance = X.corrwith(y).abs().sort_values(ascending=False)

# Split del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizzazione delle features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvataggio scaler e label encoders
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Definizione modelli di regressione
print("\n=== TRAINING E VALUTAZIONE MODELLI ===")

models_config = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'use_scaled': False
    },
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'use_scaled': True
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        },
        'use_scaled': True
    }
}

# Training e valutazione modelli
print("\n=== TRAINING E VALUTAZIONE MODELLI ===")

best_models = {}
best_configs = {}
cv_scores = {}
results = {}

for model_name, config in models_config.items():
    print(f"\n--- Processando {model_name} ---")
    
    # Selezione dati (scaled o non scaled)
    X_train_model = X_train_scaled if config['use_scaled'] else X_train
    X_test_model = X_test_scaled if config['use_scaled'] else X_test
    
    if config['params']:
        # GridSearchCV per ottimizzazione iperparametri
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_model, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = -grid_search.best_score_  # Converte da negativo a positivo
        
        # Nested Cross-Validation per valutazione più robusta
        nested_cv_scores = cross_val_score(grid_search, X_train_model, y_train, 
                                         cv=5, scoring='neg_mean_squared_error')
        nested_cv_scores = -nested_cv_scores  # Converte da negativo a positivo
        nested_cv_mean = nested_cv_scores.mean()
        nested_cv_std = nested_cv_scores.std()
        
    else:
        # Modello senza iperparametri da ottimizzare
        best_model = config['model']
        best_model.fit(X_train_model, y_train)
        best_params = {}
        
        # Cross-validation manuale
        cv_scores_manual = cross_val_score(best_model, X_train_model, y_train, 
                                         cv=5, scoring='neg_mean_squared_error')
        cv_scores_manual = -cv_scores_manual  # Converte da negativo a positivo
        best_cv_score = cv_scores_manual.mean()
        nested_cv_mean = cv_scores_manual.mean()
        nested_cv_std = cv_scores_manual.std()
    
    # Predizioni sul test set
    y_pred = best_model.predict(X_test_model)
    
    # Calcolo metriche
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Salvataggio risultati
    best_models[model_name] = best_model
    best_configs[model_name] = best_params
    cv_scores[model_name] = best_cv_score
    
    results[model_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_MSE': best_cv_score,
        'Nested_CV_Mean': nested_cv_mean,
        'Nested_CV_Std': nested_cv_std,
        'use_scaled': config['use_scaled']
    }
    
    print(f"Migliori parametri: {best_params}")
    print(f"CV MSE: {best_cv_score:.2f}")
    print(f"Nested CV MSE: {nested_cv_mean:.2f} ± {nested_cv_std:.2f}")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R²: {r2:.4f}")

# Salvataggio modelli e risultati
with open('best_regression_models.pkl', 'wb') as f:
    pickle.dump(best_models, f)
with open('regression_configs.pkl', 'wb') as f:
    pickle.dump(best_configs, f)
with open('regression_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nModelli e risultati salvati!")

# Creazione DataFrame per confronto
print("\n=== CONFRONTO RISULTATI ===")
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

# Converti le colonne numeriche al tipo corretto
for col in ['MSE', 'RMSE', 'MAE', 'R2', 'CV_MSE', 'Nested_CV_Mean', 'Nested_CV_Std']:
    results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

print(results_df)

# Salvataggio tabella risultati
results_df.to_csv('regression_comparison_results.csv')

# Visualizzazione confronto metriche
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ['MSE', 'RMSE', 'MAE', 'R2']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = results_df[metric].values
    bars = ax.bar(results_df.index, values, color=colors[i], alpha=0.7, edgecolor='black')
    ax.set_title(f'Confronto {metric}')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Aggiungi valori sulle barre
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('regression_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Analisi predizioni vs valori reali per il miglior modello
best_model_name = results_df['R2'].idxmax()
print(f"\nMiglior modello per R²: {best_model_name} (R² = {results_df.loc[best_model_name, 'R2']:.4f})")

best_model = best_models[best_model_name]
use_scaled = results[best_model_name]['use_scaled']
X_test_best = X_test_scaled if use_scaled else X_test
y_pred_best = best_model.predict(X_test_best)

# Plot predizioni vs valori reali
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_best, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valori Reali')
plt.ylabel('Predizioni')
plt.title(f'Predizioni vs Reali - {best_model_name}')
plt.grid(True, alpha=0.3)

# Residui
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predizioni')
plt.ylabel('Residui')
plt.title(f'Residui - {best_model_name}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'best_model_analysis_{best_model_name.lower()}.png', dpi=300, bbox_inches='tight')
plt.close()

# Learning curves per i modelli più performanti
print("\n=== LEARNING CURVES ===")

top_models = results_df.nlargest(3, 'R2').index.tolist()

def plot_learning_curve_regression(estimator, title, X, y, cv=5, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, 
        scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Converte MSE negative in positive
    train_scores = -train_scores
    test_scores = -test_scores
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Stampa statistiche finali  
    print(f"\n=== Learning Curve Stats - {title} Regressor ===")
    print(f"Training MSE: {train_scores_mean[-1]:.2f} ± {train_scores_std[-1]:.2f}")
    print(f"Validation MSE: {test_scores_mean[-1]:.2f} ± {test_scores_std[-1]:.2f}")
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training MSE")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation MSE")
    
    plt.xlabel("Training Examples")
    plt.ylabel("MSE")
    plt.title(f'Learning Curve - {title}')
    plt.legend(loc="best")
    
    # Analisi overfitting/underfitting
    final_train_mse = train_scores_mean[-1]
    final_val_mse = test_scores_mean[-1]
    gap = final_val_mse - final_train_mse
    
    if gap > final_train_mse * 0.5:
        status = "Possibile Overfitting"
        color = 'red'
    elif final_val_mse > np.mean(y) * 0.3:
        status = "Possibile Underfitting"
        color = 'orange'
    else:
        status = "Buon Bilanciamento"
        color = 'green'
    
    plt.text(0.05, 0.95, f"{status}\nGap: {gap:.2f}", 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'learning_curve_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curve salvata: learning_curve_{title.lower().replace(' ', '_')}.png")

for model_name in top_models:
    model = best_models[model_name]
    use_scaled = results[model_name]['use_scaled']
    X_train_model = X_train_scaled if use_scaled else X_train
    
    plot_learning_curve_regression(model, model_name, X_train_model, y_train)

# Feature importance per Random Forest
if 'RandomForest' in best_models:
    print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
    rf_model = best_models['RandomForest']
    feature_importance_rf = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 features più importanti:")
    print(feature_importance_rf.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_rf.head(15)
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importanza')
    plt.title('Feature Importance - Random Forest')
    plt.gca().invert_yaxis()
    
    # Aggiungi valori alle barre
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n=== RIEPILOGO FINALE ===")
print("Migliori modelli per ciascuna metrica:")
for metric in ['R2', 'MSE', 'MAE']:
    if metric == 'R2':
        best = results_df[metric].idxmax()
        value = results_df.loc[best, metric]
    else:
        best = results_df[metric].idxmin()
        value = results_df.loc[best, metric]
    print(f"{metric}: {best} ({value:.4f})")

print(f"\nTutti i file sono stati salvati:")
print("- best_regression_models.pkl: Modelli addestrati")
print("- regression_configs.pkl: Configurazioni ottimali")
print("- regression_results.pkl: Risultati dettagliati")
print("- regression_comparison_results.csv: Tabella comparativa")
print("- scaler.pkl: Standardizzatore per preprocessing")
print("- label_encoders.pkl: Encoder per variabili categoriche")
print("- Grafici salvati come PNG")
