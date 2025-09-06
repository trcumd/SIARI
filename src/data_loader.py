"""
Data Loader per la gestione e il caricamento dei dati.

Questo modulo fornisce funzionalità per:
- Caricamento dei dati iniziali della Knowledge Base
- Conversione tra formati RDF e tabellari
- Validazione e pulizia dei dati
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import logging
import random
from datetime import datetime

class DataLoader:
    """
    Gestisce il caricamento e la preparazione dei dati per il sistema.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Inizializza il data loader.
        
        Args:
            data_dir: Directory contenente i file di dati
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Dizionari per mappare categorie e proprietà
        self.categoria_mapping = {
            'Vegetale': ['pomodoro', 'basilico', 'aglio', 'cipolla', 'carota', 'sedano', 'peperone'],
            'Frutta': ['limone', 'arancia', 'mela', 'banana', 'fragola', 'pesca'],
            'Carne': ['manzo', 'pollo', 'maiale', 'agnello', 'tacchino'],
            'Latticini': ['latte', 'formaggio', 'burro', 'panna', 'yogurt', 'mozzarella', 'parmigiano'],
            'Cereali': ['pasta', 'riso', 'pane', 'farina', 'avena', 'orzo']
        }
        
        self.cucina_italiana = [
            'italiana', 'mediterranea', 'toscana', 'siciliana', 'napoletana', 
            'romana', 'milanese', 'piemontese'
        ]
    
    def generate_initial_kb(self) -> Dict[str, Any]:
        """
        Genera una Knowledge Base iniziale SOLO con ingredienti e ricette REALI dai file CSV.
        NON genera dati artificiali o di esempio.
        
        Returns:
            Dizionario con la struttura della KB popolata SOLO con dati reali dai CSV
        """
        kb_data = {
            "ingredienti": {},
            "ricette": {}
        }
        
        try:
            # === CARICA INGREDIENTI REALI DAL CSV ===
            ingredients_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ingredienti_reali.csv')
            
            try:
                df_ingredients = pd.read_csv(ingredients_file, on_bad_lines='skip')
                self.logger.info(f"CSV ingredienti caricato con {len(df_ingredients)} righe")
            except Exception as csv_error:
                self.logger.warning(f"Errore nel parsing CSV ingredienti: {csv_error}")
                try:
                    df_ingredients = pd.read_csv(ingredients_file, on_bad_lines='skip', quoting=3)
                    self.logger.info(f"CSV ingredienti caricato (modalità recupero) con {len(df_ingredients)} righe")
                except Exception as e:
                    self.logger.error(f"Impossibile caricare CSV ingredienti: {e}")
                    return kb_data  # Ritorna KB vuota invece di dati fittizi
            
            # Rimuovi righe con valori NaN critici
            df_ingredients = df_ingredients.dropna(subset=['nome', 'categoriaAlimentare'])
            
            # Carica SOLO gli ingredienti dal CSV usando il nome come chiave
            for idx, row in df_ingredients.iterrows():
                nome_ingrediente = row['nome'].strip()
                
                # Gestisci valori NaN con defaults ragionevoli
                peso_medio = row['peso_medio_g'] if pd.notna(row['peso_medio_g']) else 100.0
                durata_cons = row['durata_conservazione_giorni'] if pd.notna(row['durata_conservazione_giorni']) else 7
                prezzo = row['prezzo_per_kg'] if pd.notna(row['prezzo_per_kg']) else 5.0
                contenuto_acqua = row['contenuto_acqua_perc'] if pd.notna(row['contenuto_acqua_perc']) else 85.0
                calorie = row['densita_nutrizionale_kcal100g'] if pd.notna(row['densita_nutrizionale_kcal100g']) else 50.0
                indice_glic = row['indice_glicemico'] if pd.notna(row['indice_glicemico']) else 30.0
                
                kb_data["ingredienti"][nome_ingrediente] = {
                    "nome": nome_ingrediente,
                    "categoriaAlimentare": row['categoriaAlimentare'],
                    "colore": row['colore'] if pd.notna(row['colore']) else 'neutro',
                    "consistenza": row['consistenza'] if pd.notna(row['consistenza']) else 'solido',
                    "origine": row['origine'] if pd.notna(row['origine']) else 'vegetale',
                    "stagionalita": row['stagionalita'] if pd.notna(row['stagionalita']) else 'tutto_anno',
                    "provenienza_geografica": row['provenienza_geografica'] if pd.notna(row['provenienza_geografica']) else 'locale',
                    "metodo_produzione": row['metodo_produzione'] if pd.notna(row['metodo_produzione']) else 'tradizionale',
                    "livello_trasformazione": row['livello_trasformazione'] if pd.notna(row['livello_trasformazione']) else 'minimamente_lavorato',
                    "peso_medio_g": float(peso_medio),
                    "durata_conservazione_giorni": int(durata_cons),
                    "prezzo_per_kg": float(prezzo),
                    "contenuto_acqua_perc": float(contenuto_acqua),
                    "caloriePer100g": float(calorie),
                    "indice_glicemico": float(indice_glic),
                    "proteine": self._estimate_protein_content(row['categoriaAlimentare'], float(calorie)),
                    "carboidrati": self._estimate_carb_content(row['categoriaAlimentare'], float(calorie)),
                    "isAlergenico": row['categoriaAlimentare'] in ['Latticini', 'Cereali'] or 'glutine' in nome_ingrediente.lower()
                }
            
            # === CARICA RICETTE REALI DAL CSV ===
            recipes_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ricette_reali.csv')
            try:
                df_recipes = pd.read_csv(recipes_file)
                self.logger.info(f"CSV ricette caricato con {len(df_recipes)} righe")
            except Exception as e:
                self.logger.error(f"Errore nel caricamento ricette: {e}")
                return kb_data
            
            # Carica SOLO le ricette dal CSV usando il nome come chiave
            for idx, row in df_recipes.iterrows():
                nome_ricetta = row['nome'].strip()
                
                # Parsing degli ingredienti dalla colonna ingredienti
                ingredienti_lista = []
                if pd.notna(row['ingredienti']):
                    ingredienti_raw = str(row['ingredienti']).split(';')
                    for ing in ingredienti_raw:
                        ing_clean = ing.strip()
                        # Verifica che l'ingrediente esista nella KB
                        if ing_clean in kb_data["ingredienti"]:
                            ingredienti_lista.append(ing_clean)
                        else:
                            self.logger.warning(f"Ingrediente '{ing_clean}' non trovato nella KB per ricetta '{nome_ricetta}'")
                
                kb_data["ricette"][nome_ricetta] = {
                    "nome": nome_ricetta,
                    "tipo_cucina": row['tipo_cucina'] if pd.notna(row['tipo_cucina']) else 'italiana',
                    "difficolta": row['difficolta'] if pd.notna(row['difficolta']) else 'media',
                    "tipo_piatto": row['tipo_piatto'] if pd.notna(row['tipo_piatto']) else 'primo',
                    "metodo_cottura": row['metodo_cottura'] if pd.notna(row['metodo_cottura']) else 'lessato',
                    "stagionalita": row['stagionalita'] if pd.notna(row['stagionalita']) else 'tutto_anno',
                    "dieta_speciale": row['dieta_speciale'] if pd.notna(row['dieta_speciale']) else 'normale',
                    "occasione_consumo": row['occasione_consumo'] if pd.notna(row['occasione_consumo']) else 'quotidiana',
                    "numero_ingredienti": int(row['numero_ingredienti']) if pd.notna(row['numero_ingredienti']) else len(ingredienti_lista),
                    "numero_porzioni": int(row['numero_porzioni']) if pd.notna(row['numero_porzioni']) else 4,
                    "tempo_preparazione_min": int(row['tempo_preparazione_min']) if pd.notna(row['tempo_preparazione_min']) else 30,
                    "tempo_cottura_min": int(row['tempo_cottura_min']) if pd.notna(row['tempo_cottura_min']) else 15,
                    "costo_stimato_euro": float(row['costo_stimato_euro']) if pd.notna(row['costo_stimato_euro']) else 10.0,
                    "rating_medio": float(row['rating_medio']) if pd.notna(row['rating_medio']) else 4.0,
                    "numero_preparazioni": int(row['numero_preparazioni']) if pd.notna(row['numero_preparazioni']) else 1000,
                    "calorie_per_porzione": int(row['calorie_per_porzione']) if pd.notna(row['calorie_per_porzione']) else 300,
                    "tempo_totale_min": int(row['tempo_totale_min']) if pd.notna(row['tempo_totale_min']) else 45,
                    "ingredienti": ingredienti_lista
                }
            
            self.logger.info(f"KB generata con {len(kb_data['ingredienti'])} ingredienti e {len(kb_data['ricette'])} ricette")
            return kb_data
        
        except Exception as e:
            self.logger.error(f"Errore nella generazione KB: {e}")
            return {"ingredienti": {}, "ricette": {}}

    def _assegna_ingredienti_realistici(self, nome_ricetta: str, tipo_piatto: str, num_ingredienti: int, ingredienti_disponibili: List[str]) -> List[str]:
        """
        Assegna ingredienti realistici basati sul nome e tipo della ricetta.
        """
        ingredienti_assegnati = []
        
        # Ingredienti base comuni (ora usando nomi reali)
        ingredienti_base = ["Pomodoro San Marzano", "Parmigiano Reggiano 24 mesi", "Olio Extra Vergine Toscano"]
        
        # Aggiungi ingredienti specifici basati sul nome della ricetta
        if "carbonara" in nome_ricetta.lower():
            ingredienti_specifici = ["Spaghetti", "Uova", "Parmigiano Reggiano 24 mesi", "Guanciale", "Pepe nero"]
        elif "pizza" in nome_ricetta.lower():
            ingredienti_specifici = ["Pomodoro San Marzano", "Mozzarella di Bufala Campana", "Basilico Genovese DOP"]
        elif "risotto" in nome_ricetta.lower():
            ingredienti_specifici = ["Risotto Carnaroli", "Brodo di Carne", "Parmigiano Reggiano 24 mesi"]
        elif "pasta" in nome_ricetta.lower():
            ingredienti_specifici = ["Pomodoro San Marzano", "Olio Extra Vergine Toscano", "Parmigiano Reggiano 24 mesi"]
        else:
            # Selezione casuale per ricette non specifiche
            ingredienti_specifici = random.sample(ingredienti_disponibili, min(num_ingredienti, len(ingredienti_disponibili)))
        
        # Combina ingredienti base e specifici
        ingredienti_assegnati = list(set(ingredienti_base + ingredienti_specifici))
        
        # Assicurati di avere il numero corretto di ingredienti
        while len(ingredienti_assegnati) < num_ingredienti and len(ingredienti_assegnati) < len(ingredienti_disponibili):
            candidato = random.choice(ingredienti_disponibili)
            if candidato not in ingredienti_assegnati:
                ingredienti_assegnati.append(candidato)
        
        return ingredienti_assegnati[:num_ingredienti]
    
    def _generate_synthetic_kb(self) -> Dict[str, Any]:
        """
        Metodo di fallback per generare KB sintetica se i file reali non sono disponibili.
        """
        kb_data = {
            "ingredienti": {},
            "ricette": {},
            "utensili": {},
            "tipiCucina": {}
        }
        
        # Genera ingredienti base
        for i, categoria in enumerate(['Vegetale', 'Carne', 'Latticini'], 1):
            kb_data["ingredienti"][f"ingrediente_{i}"] = {
                "nome": f"Ingrediente {i}",
                "categoriaAlimentare": categoria,
                "caloriePer100g": 100,
                "isAlergenico": False
            }
        
        # Genera ricette base
        kb_data["ricette"]["ricetta_1"] = {
            "nome": "Ricetta Base",
            "ingredienti": ["ingrediente_1"],
            "tempoDiPreparazione": 30,
            "porzioni": 4,
            "caloriePerPorzione": 200
        }
        
        return kb_data
    
    def save_initial_kb(self, kb_data: Dict[str, Any], filename: str = "initial_kb.json") -> bool:
        """
        Salva la Knowledge Base iniziale in un file JSON.
        
        Args:
            kb_data: Dati della Knowledge Base
            filename: Nome del file di output
            
        Returns:
            True se il salvataggio è avvenuto con successo
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            os.makedirs(self.data_dir, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(kb_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Knowledge Base salvata in {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio della KB: {e}")
            return False
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida i dati e restituisce statistiche di qualità.
        
        Args:
            df: DataFrame da validare
            
        Returns:
            Dizionario con le statistiche di validazione
        """
        stats = {
            'num_samples': len(df),
            'num_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Verifica feature categoriche
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            stats[f'{col}_unique_values'] = df[col].nunique()
            stats[f'{col}_most_frequent'] = df[col].mode().iloc[0] if not df[col].empty else None
        
        # Verifica feature numeriche
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_std'] = df[col].std()
            stats[f'{col}_min'] = df[col].min()
            stats[f'{col}_max'] = df[col].max()
        
        # Verifica distribuzione del target
        if 'categoriaAlimentare' in df.columns:
            target_dist = df['categoriaAlimentare'].value_counts()
            stats['target_distribution'] = target_dist.to_dict()
            stats['target_balance'] = target_dist.min() / target_dist.max()
        
        self.logger.info(f"Validazione completata: {stats['num_samples']} campioni, "
                        f"{stats['duplicates']} duplicati, "
                        f"{sum(stats['missing_values'].values())} valori mancanti")
        
        return stats
    
    def create_test_recipes(self) -> List[Dict[str, Any]]:
        """
        Crea un set di ricette di test per validare il sistema.
        
        Returns:
            Lista di ricette di test
        """
        test_recipes = [
            {
                "nome": "Pizza Margherita",
                "tipo_cucina": "italiana",
                "difficolta": "media",
                "tipo_piatto": "primo",
                "metodo_cottura": "al_forno",
                "num_ingredienti": 6,
                "porzioni": 4,
                "is_vegetariano": True,
                "ingredienti_stimati": ["farina", "pomodoro", "mozzarella", "basilico", "olio", "sale"]
            },
            {
                "nome": "Salmone alla Griglia",
                "tipo_cucina": "mediterranea",
                "difficolta": "facile",
                "tipo_piatto": "secondo",
                "metodo_cottura": "alla_griglia",
                "num_ingredienti": 4,
                "porzioni": 2,
                "is_vegetariano": False,
                "ingredienti_stimati": ["salmone", "limone", "olio", "sale"]
            },
            {
                "nome": "Tiramisù",
                "tipo_cucina": "italiana",
                "difficolta": "difficile",
                "tipo_piatto": "dolce",
                "metodo_cottura": "crudo",
                "num_ingredienti": 8,
                "porzioni": 6,
                "is_vegetariano": True,
                "ingredienti_stimati": ["mascarpone", "uova", "caffè", "savoiardi", "cacao", "zucchero", "liquore", "panna"]
            }
        ]
        
        return test_recipes
    
    def export_data_summary(self) -> Dict[str, Any]:
        """
        Esporta un riepilogo di tutti i dati disponibili.
        
        Returns:
            Dizionario con il riepilogo dei dati
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': self.data_dir,
            'available_files': [],
            'categories_mapping': self.categoria_mapping,
            'cuisine_types': self.cucina_italiana
        }
        
        # Lista dei file disponibili
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, file)
                if os.path.isfile(file_path):
                    file_info = {
                        'name': file,
                        'size_bytes': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    }
                    summary['available_files'].append(file_info)
        
        return summary
    
    def _estimate_protein_content(self, categoria: str, calorie: float) -> float:
        """
        Stima il contenuto proteico basato sulla categoria alimentare e le calorie.
        
        Args:
            categoria: Categoria alimentare dell'ingrediente
            calorie: Calorie per 100g
            
        Returns:
            Contenuto proteico stimato in grammi per 100g
        """
        if categoria == 'Carne':
            return calorie * 0.12  # ~20-25g proteine per 100g
        elif categoria == 'Latticini':
            return calorie * 0.08  # ~10-15g proteine per 100g
        elif categoria == 'Cereali':
            return calorie * 0.03  # ~8-12g proteine per 100g
        elif categoria == 'Vegetale':
            return calorie * 0.05  # ~2-4g proteine per 100g
        else:
            return calorie * 0.02  # Default basso
    
    def _estimate_carb_content(self, categoria: str, calorie: float) -> float:
        """
        Stima il contenuto di carboidrati basato sulla categoria alimentare e le calorie.
        
        Args:
            categoria: Categoria alimentare dell'ingrediente
            calorie: Calorie per 100g
            
        Returns:
            Contenuto di carboidrati stimato in grammi per 100g
        """
        if categoria == 'Cereali':
            return calorie * 0.18  # ~60-80g carboidrati per 100g
        elif categoria == 'Vegetale':
            return calorie * 0.15  # Varia molto, stima media
        elif categoria == 'Latticini':
            return calorie * 0.02  # ~3-5g carboidrati per 100g
        elif categoria == 'Carne':
            return 0.0  # Praticamente zero carboidrati
        else:
            return calorie * 0.05  # Default moderato
    
    def _create_fallback_ingredients(self):
        """Crea un set minimo di ingredienti di fallback in caso di errore CSV"""
        fallback_data = {
            'nome': ['Pomodoro', 'Mozzarella', 'Basilico'],
            'categoriaAlimentare': ['Vegetale', 'Latticini', 'Erbe'],
            'colore': ['rosso', 'bianco', 'verde'],
            'consistenza': ['solido', 'solido', 'foglia'],
            'origine': ['vegetale', 'animale', 'vegetale'],
            'stagionalita': ['estate', 'tutto_anno', 'estate'],
            'provenienza': ['italiana', 'italiana', 'italiana'],
            'preparazione': ['fresco', 'fresco', 'fresco'],
            'conservazione': ['fresco', 'fresco', 'fresco'],
            'prezzo_base': [2.0, 8.0, 3.0],
            'prezzo_kg': [3.0, 20.0, 15.0],
            'calorie_100g': [18.0, 280.0, 25.0],
            'carboidrati_100g': [3.9, 2.3, 2.6],
            'proteine_100g': [0.9, 18.0, 3.2],
            'grassi_100g': [0.2, 22.0, 0.6]
        }
        return pd.DataFrame(fallback_data)
