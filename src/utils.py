"""
Modulo di utilità per funzioni di supporto comuni.

Questo modulo fornisce funzioni di supporto per:
- Logging e configurazione
- Validazione dati
- Conversioni di formato
- Funzioni matematiche ausiliarie
- Visualizzazioni
"""

import logging
import os
import sys
import json
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import re

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura il sistema di logging per l'applicazione.
    
    Args:
        log_level: Livello di logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Percorso del file di log (opzionale)
        
    Returns:
        Logger configurato
    """
    # Converti il livello in numero
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Livello di log non valido: {log_level}')
    
    # Formato del log
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configurazione base
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Logger principale
    logger = logging.getLogger('RecipeKB')
    
    # Handler per file se specificato
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Handler per console con colori
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Formato colorato per la console
    class ColoredFormatter(logging.Formatter):
        """Formatter con colori per diversi livelli di log."""
        
        COLORS = {
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[35m',  # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            log_color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{log_color}{record.levelname}{self.RESET}"
            return super().format(record)
    
    console_formatter = ColoredFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging configurato con livello {log_level}")
    return logger

def validate_json_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Valida la struttura di un dizionario JSON.
    
    Args:
        data: Dizionario da validare
        required_keys: Lista delle chiavi richieste
        
    Returns:
        True se la struttura è valida, False altrimenti
    """
    try:
        for key in required_keys:
            if key not in data:
                return False
        return True
    except (TypeError, AttributeError):
        return False

def clean_string(text: str) -> str:
    """
    Pulisce una stringa rimuovendo caratteri speciali e normalizzando.
    
    Args:
        text: Stringa da pulire
        
    Returns:
        Stringa pulita
    """
    if not isinstance(text, str):
        return str(text)
    
    # Rimuovi caratteri speciali e normalizza
    cleaned = re.sub(r'[^\w\s-]', '', text)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    cleaned = cleaned.lower()
    
    return cleaned

def generate_id(prefix: str, counter: int) -> str:
    """
    Genera un ID univoco con prefisso e contatore.
    
    Args:
        prefix: Prefisso per l'ID
        counter: Numero progressivo
        
    Returns:
        ID generato
    """
    return f"{prefix}_{counter:04d}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divisione sicura che gestisce la divisione per zero.
    
    Args:
        numerator: Numeratore
        denominator: Denominatore
        default: Valore di default se il denominatore è zero
        
    Returns:
        Risultato della divisione o valore di default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formatta un valore come percentuale.
    
    Args:
        value: Valore da formattare (0.0 - 1.0)
        decimals: Numero di decimali
        
    Returns:
        Stringa formattata come percentuale
    """
    try:
        return f"{value * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"

def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calcola l'intervallo di confidenza per una lista di valori.
    
    Args:
        values: Lista di valori numerici
        confidence: Livello di confidenza (0.0 - 1.0)
        
    Returns:
        Dizionario con mean, lower_bound, upper_bound
    """
    if not values:
        return {'mean': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0}
    
    try:
        import numpy as np
        from scipy import stats
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        std_err = stats.sem(values_array)
        h = std_err * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        
        return {
            'mean': float(mean),
            'lower_bound': float(mean - h),
            'upper_bound': float(mean + h)
        }
    except ImportError:
        # Fallback senza scipy
        mean = sum(values) / len(values)
        std = (sum([(x - mean) ** 2 for x in values]) / (len(values) - 1)) ** 0.5
        margin = 1.96 * std / (len(values) ** 0.5)  # Approssimazione per 95%
        
        return {
            'mean': mean,
            'lower_bound': mean - margin,
            'upper_bound': mean + margin
        }

class Timer:
    """Context manager per misurare il tempo di esecuzione."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Iniziato: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.elapsed_time = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"Completato: {self.description} in {self.elapsed_time:.3f} secondi")

class ProgressTracker:
    """Tracker per monitorare il progresso di operazioni lunghe."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1):
        """Aggiorna il progresso."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: N/A"
        
        self.logger.info(f"{self.description}: {self.current}/{self.total} "
                        f"({percentage:.1f}%) - {eta_str}")
    
    def finish(self):
        """Completa il tracker."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.description} completato in {elapsed:.2f} secondi")

def create_summary_table(data: Dict[str, Any], title: str = "Summary") -> str:
    """
    Crea una tabella di riepilogo formattata.
    
    Args:
        data: Dizionario con i dati da visualizzare
        title: Titolo della tabella
        
    Returns:
        Stringa con la tabella formattata
    """
    lines = [
        f"\n{'=' * 50}",
        f"{title:^50}",
        f"{'=' * 50}"
    ]
    
    for key, value in data.items():
        if isinstance(value, float):
            value_str = f"{value:.3f}"
        elif isinstance(value, dict):
            value_str = f"{len(value)} items"
        elif isinstance(value, list):
            value_str = f"{len(value)} items"
        else:
            value_str = str(value)
        
        # Formatta la riga
        key_formatted = key.replace('_', ' ').title()
        line = f"{key_formatted:<30} {value_str:>15}"
        lines.append(line)
    
    lines.append("=" * 50)
    return "\n".join(lines)

def export_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> bool:
    """
    Esporta le metriche in un file JSON.
    
    Args:
        metrics: Dizionario con le metriche
        filepath: Percorso del file di output
        
    Returns:
        True se l'esportazione è avvenuta con successo
    """
    try:
        # Aggiungi timestamp
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salva il file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Errore nell'esportazione delle metriche: {e}")
        return False

def load_config(config_path: str, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Carica un file di configurazione JSON con valori di default.
    
    Args:
        config_path: Percorso del file di configurazione
        default_config: Configurazione di default
        
    Returns:
        Dizionario con la configurazione
    """
    if default_config is None:
        default_config = {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Unisci con i default
        final_config = default_config.copy()
        final_config.update(config)
        
        return final_config
        
    except FileNotFoundError:
        logging.getLogger(__name__).warning(f"File di configurazione non trovato: {config_path}. "
                                           "Utilizzando configurazione di default.")
        return default_config
    except json.JSONDecodeError as e:
        logging.getLogger(__name__).error(f"Errore nel parsing del file di configurazione: {e}")
        return default_config

def validate_email(email: str) -> bool:
    """
    Valida un indirizzo email.
    
    Args:
        email: Indirizzo email da validare
        
    Returns:
        True se l'email è valida
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Tronca un testo se supera la lunghezza massima.
    
    Args:
        text: Testo da troncare
        max_length: Lunghezza massima
        suffix: Suffisso da aggiungere se troncato
        
    Returns:
        Testo troncato
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def batch_process(items: List[Any], batch_size: int, process_func: Callable) -> List[Any]:
    """
    Processa una lista di elementi in batch.
    
    Args:
        items: Lista di elementi da processare
        batch_size: Dimensione del batch
        process_func: Funzione di processing che accetta una lista
        
    Returns:
        Lista dei risultati
    """
    results = []
    logger = logging.getLogger(__name__)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        logger.debug(f"Processando batch {i//batch_size + 1}: {len(batch)} elementi")
        
        try:
            batch_results = process_func(batch)
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Errore nel processing del batch {i//batch_size + 1}: {e}")
    
    return results

def normalize_scores(scores: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """
    Normalizza una lista di punteggi in un range specifico.
    
    Args:
        scores: Lista di punteggi
        min_val: Valore minimo del range
        max_val: Valore massimo del range
        
    Returns:
        Lista di punteggi normalizzati
    """
    if not scores:
        return []
    
    current_min = min(scores)
    current_max = max(scores)
    
    if current_max == current_min:
        return [min_val] * len(scores)
    
    normalized = []
    for score in scores:
        norm_score = (score - current_min) / (current_max - current_min)
        norm_score = norm_score * (max_val - min_val) + min_val
        normalized.append(norm_score)
    
    return normalized

def get_system_info() -> Dict[str, Any]:
    """
    Raccoglie informazioni sul sistema.
    
    Returns:
        Dizionario con informazioni di sistema
    """
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'hostname': platform.node(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Aggiungi informazioni su memoria se possibile
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['memory_total_gb'] = round(memory.total / (1024**3), 2)
        info['memory_available_gb'] = round(memory.available / (1024**3), 2)
        info['cpu_count'] = psutil.cpu_count()
    except ImportError:
        pass
    
    return info
