# Assistente Intelligente per la Curatela di Knowledge Base, Analisi Ricette e Inferenza Semantica

## Panoramica del Progetto

Questo progetto implementa un assistente intelligente per la gestione e l'analisi di una Knowledge Base nel dominio delle ricette culinarie, integrando:

1. **Knowledge Graphs e Ontologie** - Rappresentazione strutturata della conoscenza
2. **Motore Prolog** - Inferenza semantica e query logiche
3. **Apprendimento Supervisionato** - Clustering, classificazione e regressione per analisi nutrizionale e predizione calorie

## Struttura del Progetto

```
ICON2/
├── src/
│   ├── kb_manager.py          # Gestione Knowledge Base e interfaccia Prolog
│   ├── prolog_engine.py       # Motore di inferenza Prolog
│   ├── data_loader.py         # Caricamento e gestione dati
│   ├── regressorIngredienti.py# Modelli regressivi per ingredienti
│   ├── regressorRicette.py    # Modelli regressivi per ricette
│   └── utils.py               # Funzioni di supporto
├── clustering_classification_analysis.py # Analisi clustering e classificazione
├── config.py                  # Configurazione pipeline
├── main.py                    # Interfaccia CLI principale
├── data/
│   ├── ingredienti_reali.csv  # Database ingredienti
│   ├── ricette_reali.csv      # Database ricette
│   └── kb_state.json          # Stato Knowledge Base
├── documentazione/
│   └── main.tex               # Documento LaTeX
├── logs/
│   └── recipe_assistant.log   # Log di sistema
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installazione

```bash
pip install -r requirements.txt
```

## Utilizzo

Avvia l'interfaccia CLI:

```bash
python main.py
```

### Funzionalità Principali

1. **Caricamento KB**: Carica la Knowledge Base da file CSV reali
2. **Query Prolog**: Esegui query logiche e inferenza semantica
3. **Clustering e Classificazione**: Analisi automatica delle ricette
4. **Regressione Calorica**: Predizione calorie per ricette e ingredienti
5. **Visualizzazione**: Statistiche, grafici e risultati delle analisi

## Esempi di Query

- `ingredienti_ricetta(nome_ricetta)` - Trova ingredienti di una ricetta
- `is_vegetariano(nome_ricetta)` - Verifica se una ricetta è vegetariana
- `categoria_ingrediente(nome_ingrediente)` - Trova la categoria di un ingrediente

## Testing

```bash
pytest
```

## Componenti Tecnici

### Knowledge Base & Prolog
- Rappresentazione simbolica e inferenza logica
- Query semantiche e ragionamento automatico

### Machine Learning
- Clustering K-Means per ricette
- Classificazione SVM, Random Forest
- Regressione SVR, Ridge per calorie

### Database Reale
- 115 ricette complete con ingredienti autentici
- 372 ingredienti con proprietà nutrizionali
- Database CSV validato e ottimizzato

## Documentazione

La documentazione tecnica e metodologica è disponibile in `documentazione/main.tex` (LaTeX).

---

Per dettagli su moduli, pipeline e risultati, consultare il documento LaTeX e i file di log generati.