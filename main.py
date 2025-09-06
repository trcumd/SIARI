"""
Interfaccia CLI principale per l'Assistente Intelligente per la Curatela di Knowledge Base.

Questo modulo fornisce un'interfaccia a linea di comando per:
- Caricamento e gestione della Knowledge Base
- Esecuzione di query Prolog
- Visualizzazione di risultati e statistiche
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional
import logging

# Aggiungi il percorso src al PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.kb_manager import RecipeKnowledgeBase
    from src.prolog_engine import PrologEngine, PrologQueryBuilder
    from src.data_loader import DataLoader
    from src.utils import setup_logging, create_summary_table, Timer, ProgressTracker
except ImportError as e:
    print(f"Errore nell'importazione dei moduli: {e}")
    print("Assicurati che tutti i requirements siano installati: pip install -r requirements.txt")
    sys.exit(1)

class RecipeAssistantCLI:
    """
    Interfaccia CLI per l'assistente di ricette.
    """
    
    def __init__(self):
        """Inizializza l'assistente CLI."""
        self.kb_manager = None
        self.prolog_engine = None
        self.data_loader = None
        self.logger = None
        
        # Configurazione di default
        self.config = {
            'data_dir': 'data',
            'log_level': 'INFO',
            'log_file': 'logs/recipe_assistant.log'
        }
    
    def initialize(self, log_level: str = "INFO"):
        """
        Inizializza tutti i componenti del sistema.
        
        Args:
            log_level: Livello di logging
        """
        # Setup logging
        self.logger = setup_logging(log_level, self.config['log_file'])
        self.logger.info("Inizializzazione del Recipe Assistant")
        
        # Inizializza componenti
        self.kb_manager = RecipeKnowledgeBase()
        self.data_loader = DataLoader(self.config['data_dir'])
        
        # Inizializza motore Prolog
        try:
            self.prolog_engine = PrologEngine()
            self.logger.info("Motore Prolog inizializzato")
        except ImportError:
            self.logger.error("PySwip non disponibile. Installare con: pip install pyswip")
            self.prolog_engine = None
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione Prolog: {e}")
            self.prolog_engine = None
        
        self.logger.info("Componenti inizializzati con successo")
    
    def load_knowledge_base(self) -> bool:
        """
        Carica la Knowledge Base dai dati CSV reali.
        
        Returns:
            True se il caricamento è avvenuto con successo
        """
        self._kb_loaded = False
        
        ingredients_csv = os.path.join(self.config['data_dir'], 'ingredienti_reali.csv')
        recipes_csv = os.path.join(self.config['data_dir'], 'ricette_reali.csv')
        state_file = os.path.join(self.config['data_dir'], 'kb_state.json')
        
        # Verifica se i dati sono già aggiornati
        if self._is_kb_updated(ingredients_csv, recipes_csv, state_file):
            print("📁 Knowledge Base già caricata e aggiornata!")
            
            # Anche se la KB è già caricata, dobbiamo importarla nel Prolog
            if self.prolog_engine and not hasattr(self, '_prolog_loaded'):
                print("🔄 Importazione dati nel motore Prolog...")
                
                # Carica i dati nella KB se non sono già in memoria
                if not hasattr(self, '_kb_loaded') or not self._kb_loaded:
                    kb_data = self.data_loader.generate_initial_kb()
                    self._load_kb_data_directly(kb_data)
                
                # Debug: verifica se la KB ha dati
                ingredients = self.kb_manager.get_all_ingredients()
                recipes = self.kb_manager.get_all_recipes()
                print(f"🔍 Debug: KB ha {len(ingredients)} ingredienti e {len(recipes)} ricette")
                
                self.prolog_engine.import_from_kb(self.kb_manager)
                self.prolog_engine.load_default_rules()
                self._prolog_loaded = True
                
                # Stampa statistiche Prolog
                prolog_stats = self.prolog_engine.get_stats()
                print(create_summary_table(prolog_stats, "Statistiche Prolog"))
            
            return True
        
        # Controlla che i CSV esistano
        if not (os.path.exists(ingredients_csv) and os.path.exists(recipes_csv)):
            self.logger.warning("File CSV non trovati, genero KB di default")
            print("⚠️  File CSV non trovati, genero KB di default")
            return self.generate_default_kb()
        
        with Timer("Caricamento Knowledge Base da CSV"):
            # Genera KB dai dati reali CSV  
            kb_data = self.data_loader.generate_initial_kb()
            
            # Carica direttamente nel KB manager
            self._load_kb_data_directly(kb_data)
            
            # Aggiorna lo stato
            self._update_csv_state(ingredients_csv, recipes_csv, state_file)
            self._kb_loaded = True
        
        # Importa i dati nel motore Prolog se disponibile
        if self.prolog_engine:
            self.logger.info("Importazione dati nel motore Prolog")
            
            # Debug: verifica se la KB ha dati
            ingredients = self.kb_manager.get_all_ingredients()
            recipes = self.kb_manager.get_all_recipes()
            print(f"🔍 Debug: KB ha {len(ingredients)} ingredienti e {len(recipes)} ricette")
            
            self.prolog_engine.import_from_kb(self.kb_manager)
            self.prolog_engine.load_default_rules()
        
        # Stampa statistiche
        kb_stats = self.kb_manager.get_kb_stats()
        
        print("\n✅ Knowledge Base caricata con successo!")
        print(create_summary_table(kb_stats, "Statistiche Knowledge Base"))
        
        # Statistiche Prolog se disponibile
        if self.prolog_engine:
            prolog_stats = self.prolog_engine.get_stats()
            print(create_summary_table(prolog_stats, "Statistiche Prolog"))
        
        return True
    
    def generate_default_kb(self) -> bool:
        """Genera una KB di default dai CSV se disponibili."""
        try:
            with ProgressTracker(["Caricamento CSV", "Generazione KB", "Caricamento in memoria"], 
                               "Generazione Knowledge Base") as progress:
                
                progress.update("Caricamento CSV")
                kb_data = self.data_loader.generate_initial_kb()
                
                progress.update("Generazione KB")
                
                # Carica direttamente nel KB manager senza usare file JSON
                self._load_kb_data_directly(kb_data)
                
                # Importa nel Prolog se disponibile
                if self.prolog_engine:
                    # Debug: verifica se la KB ha dati  
                    ingredients = self.kb_manager.get_all_ingredients()
                    recipes = self.kb_manager.get_all_recipes()
                    print(f"🔍 Debug: KB ha {len(ingredients)} ingredienti e {len(recipes)} ricette")
                    
                    self.prolog_engine.import_from_kb(self.kb_manager)
                    self.prolog_engine.load_default_rules()
                
                print("✅ Knowledge Base da dati reali CSV generata e caricata!")
                
                # Genera anche dati di training
                self.generate_training_data()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Errore nella generazione della KB da CSV: {e}")
            print(f"❌ Errore nella generazione della KB: {e}")
            return False
    
    def _load_kb_data_directly(self, kb_data: Dict[str, Any]):
        """
        Carica i dati della KB direttamente nel KB manager senza usare file JSON.
        
        Args:
            kb_data: Dizionario con i dati della KB
        """
        # Carica ingredienti SENZA salvarli nel CSV (stanno già nel CSV)
        if 'ingredienti' in kb_data:
            for ingredient_id, ingredient_data in kb_data['ingredienti'].items():
                self._add_ingredient_without_csv_save(ingredient_id, ingredient_data)
        
        # Carica ricette SENZA salvarle nel CSV (stanno già nel CSV)
        if 'ricette' in kb_data:
            for recipe_id, recipe_data in kb_data['ricette'].items():
                self._add_recipe_without_csv_save(recipe_id, recipe_data)
    
    def _add_ingredient_without_csv_save(self, ingredient_id: str, ingredient_data: Dict[str, Any]):
        """Aggiunge un ingrediente senza salvarlo nel CSV."""
        try:
            # Aggiungi l'ingrediente alla KB usando la signature corretta
            self.kb_manager.add_ingredient(ingredient_id, ingredient_data)
        except Exception as e:
            self.logger.warning(f"Errore aggiungendo ingrediente {ingredient_id}: {e}")
    
    def _add_recipe_without_csv_save(self, recipe_id: str, recipe_data: Dict[str, Any]):
        """Aggiunge una ricetta senza salvarla nel CSV."""
        try:
            # Aggiungi la ricetta alla KB usando la signature corretta
            self.kb_manager.add_recipe(recipe_id, recipe_data)
        except Exception as e:
            self.logger.warning(f"Errore aggiungendo ricetta {recipe_id}: {e}")
    
    def _is_kb_updated(self, ingredients_csv: str, recipes_csv: str, state_file: str) -> bool:
        """Verifica se la KB è aggiornata rispetto ai file CSV."""
        try:
            if not os.path.exists(state_file):
                return False
            
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Verifica se i timestamp dei file CSV sono cambiati
            current_ingredients_mtime = os.path.getmtime(ingredients_csv) if os.path.exists(ingredients_csv) else 0
            current_recipes_mtime = os.path.getmtime(recipes_csv) if os.path.exists(recipes_csv) else 0
            
            stored_ingredients_mtime = state.get('ingredients_mtime', 0)
            stored_recipes_mtime = state.get('recipes_mtime', 0)
            
            return (current_ingredients_mtime == stored_ingredients_mtime and 
                    current_recipes_mtime == stored_recipes_mtime)
                    
        except Exception as e:
            self.logger.warning(f"Errore nella verifica dello stato KB: {e}")
            return False
    
    def _update_csv_state(self, ingredients_csv: str, recipes_csv: str, state_file: str):
        """Aggiorna lo stato dei file CSV."""
        try:
            state = {
                'ingredients_mtime': os.path.getmtime(ingredients_csv) if os.path.exists(ingredients_csv) else 0,
                'recipes_mtime': os.path.getmtime(recipes_csv) if os.path.exists(recipes_csv) else 0,
                'last_updated': time.time()
            }
            
            # Crea la directory se non esiste
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Impossibile aggiornare lo stato CSV: {e}")
    
    def execute_prolog_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Esegue una query Prolog.
        
        Args:
            query: Query da eseguire
            
        Returns:
            Lista dei risultati
        """
        if not self.prolog_engine:
            print("❌ Motore Prolog non disponibile")
            return []
            
        try:
            with Timer(f"Esecuzione query Prolog: {query}"):
                results = self.prolog_engine.query(query)
            
            print(f"\n🔍 Query Prolog: {query}")
            
            if results:
                print(f"✅ Trovati {len(results)} risultati:")
                for i, result in enumerate(results, 1):
                    if result:
                        substitutions = ", ".join([f"{var}={val}" for var, val in result.items()])
                        print(f"  {i}. {substitutions}")
                    else:
                        print(f"  {i}. ✓ (query soddisfatta)")
            else:
                print("❌ Nessun risultato trovato")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione della query Prolog: {e}")
            print(f"❌ Errore nell'esecuzione della query: {e}")
            return []
    
    def show_prolog_examples(self):
        """Mostra esempi di query Prolog avanzate."""
        if not self.prolog_engine:
            print("❌ Motore Prolog non disponibile")
            return
        
        print("\n🎯 ESEMPI DI QUERY PROLOG AVANZATE:")
        print("="*50)
        
        examples = [
            ("Ricette italiane", "ricetta(R), tipo_cucina(R, italiana)"),
            ("Ricette vegetariane", "ricetta_vegetariana(R)"),
            ("Ricette veloci (< 30 min)", "ricetta_veloce(R)"),
            ("Ricette facili", "ricetta_facile(R)"),
            ("Primi piatti", "primi_piatti(R)"),
            ("Ricette economiche", "ricetta_economica(R)"),
            ("Ricette salutari", "ricetta_salutare(R)"),
            ("Query complessa", "ricetta_vegetariana(R), ricetta_veloce(R), not(contiene_allergenici(R))")
        ]
        
        for desc, query in examples:
            print(f"\n📋 {desc}:")
            print(f"   Query: {query}")
            
            try:
                results = self.prolog_engine.query(query)
                if results and len(results) > 0:
                    print(f"   ✅ {len(results)} risultati trovati")
                else:
                    print("   ❌ Nessun risultato")
            except Exception as e:
                print(f"   ❌ Errore: {e}")
        
        print("\n💡 Suggerimento: Copia una delle query sopra per provarla!")
    
    def generate_training_data(self):
        """Genera dati di training per modelli ML."""
        try:
            print("🤖 Generazione dati di training...")
            
            # Qui potresti aggiungere logica per generare dati di training
            # Per ora, salviamo solo un file di esempio
            
            training_data = {
                'version': '1.0',
                'generated_at': time.time(),
                'description': 'Dati di training generati dalla Knowledge Base'
            }
            
            training_file = os.path.join(self.config['data_dir'], 'training_data.json')
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Dati di training salvati in: {training_file}")
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione dati training: {e}")
            print(f"❌ Errore: {e}")
    
    def run_interactive_mode(self):
        """Modalità interattiva CLI."""
        print("\n🎯 Modalità Interattiva Recipe Assistant")
        print("=" * 50)
        
        while True:
            print("\n📋 MENU PRINCIPALE")
            print("-" * 30)
            print("1. Mostra statistiche KB")
            print("2. Query Prolog")
            print("3. Esempi di query")
            print("4. Esci")
            
            choice = input("\nScegli (1-4): ").strip()
            
            if choice == '1':
                if self.kb_manager:
                    stats = self.kb_manager.get_kb_stats()
                    print("\n📊 Statistiche Knowledge Base:")
                    print(create_summary_table(stats, "Knowledge Base"))
                    
                    if self.prolog_engine:
                        prolog_stats = self.prolog_engine.get_stats()
                        print(create_summary_table(prolog_stats, "Motore Prolog"))
                else:
                    print("❌ Knowledge Base non disponibile")
                    
            elif choice == '2':
                self._prolog_query_menu()
                
            elif choice == '3':
                self.show_prolog_examples()
                
            elif choice == '4':
                print("👋 Arrivederci!")
                break
                
            else:
                print("❌ Opzione non valida")
    
    def _prolog_query_menu(self):
        """Menu per query Prolog avanzate."""
        if not self.prolog_engine:
            print("❌ Motore Prolog non disponibile")
            return
        
        print("\n🔍 QUERY PROLOG AVANZATE")
        print("-" * 40)
        print("1. Query Prolog personalizzata")
        print("2. Ricette vegetariane")
        print("3. Ricette veloci (< 30 min)")
        print("4. Ricette light")
        print("5. Query complessa (vegetariana + veloce + senza allergeni)")
        print("6. Torna al menu principale")
        
        while True:
            choice = input("\nScegli (1-6): ").strip()
            
            if choice == '1':
                query = input("Inserisci la query Prolog: ").strip()
                if query:
                    self.execute_prolog_query(query)
                else:
                    print("❌ Query vuota")
                break
                    
            elif choice == '2':
                self.execute_prolog_query(PrologQueryBuilder.find_vegetarian_recipes())
                break
                
            elif choice == '3':
                self.execute_prolog_query(PrologQueryBuilder.find_quick_recipes())
                break
                
            elif choice == '4':
                self.execute_prolog_query(PrologQueryBuilder.find_light_recipes())
                break
                
            elif choice == '5':
                self.execute_prolog_query(PrologQueryBuilder.complex_query_example())
                break
                
            elif choice == '6':
                break
                
            else:
                print("❌ Opzione non valida")

def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description='Recipe Assistant CLI')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Livello di logging')
    parser.add_argument('--data-dir', default='data',
                       help='Directory dei dati')
    
    args = parser.parse_args()
    
    # Inizializza l'assistente
    assistant = RecipeAssistantCLI()
    assistant.config['data_dir'] = args.data_dir
    assistant.initialize(args.log_level)
    
    # Carica automaticamente la Knowledge Base all'avvio
    print("🔄 Caricamento automatico della Knowledge Base...")
    if not assistant.load_knowledge_base():
        print("❌ Errore nel caricamento automatico della Knowledge Base")
        print("Il sistema continuerà con funzionalità limitate")
    
    try:
        # Avvia modalità interattiva
        assistant.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrotto dall'utente. Arrivederci!")
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
