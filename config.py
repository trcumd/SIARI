# Configurazione per l'Assistente Intelligente di Ricette

# Percorsi
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Impostazioni Knowledge Base
KB_FILE = "data/initial_kb.json"
ONTOLOGY_FILE = "data/ontology/recipe_ontology.ttl"

# Impostazioni Datalog
DEFAULT_RULES_ENABLED = True
BOTTOM_UP_MAX_ITERATIONS = 100

CALORIE_REGRESSOR_CONFIG = {
    "regularization": "ridge", 
    "alpha": 1.0
}

# Impostazioni logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Generazione dati
TRAINING_SAMPLES_INGREDIENTS = 200
TRAINING_SAMPLES_RECIPES = 150

# Performance
BATCH_SIZE = 32
MAX_ITERATIONS = 1000
