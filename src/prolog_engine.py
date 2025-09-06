"""
Motore Prolog per il ragionamento logico e l'inferenza semantica avanzata.

Questo modulo implementa un motore di inferenza Prolog con supporto per:
- Query congiunte native (AND)
- Regole complesse con aritmetica
- Strutture dati avanzate
- Predicati built-in
- Ottimizzazioni con cut operator
"""

import logging
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass
import re

try:
    from pyswip import Prolog
    PYSWIP_AVAILABLE = True
except ImportError:
    PYSWIP_AVAILABLE = False
    print("⚠️  PySwip non disponibile. Installare con: pip install pyswip")

@dataclass
class PrologFact:
    """Rappresenta un fatto Prolog."""
    predicate: str
    terms: List[str]
    
    def __str__(self):
        terms_str = ", ".join(str(term) for term in self.terms)
        return f"{self.predicate}({terms_str})"
    
    def to_prolog_string(self):
        """Converte il fatto in stringa Prolog valida."""
        if not self.terms:
            return f"{self.predicate}"
        
        # Gestisce stringhe e numeri appropriatamente
        formatted_terms = []
        for term in self.terms:
            if isinstance(term, str):
                # Escape virgolette singole all'interno delle stringhe
                escaped_term = term.replace("'", "\\'")
                # Wrap sempre in virgolette per sicurezza
                formatted_terms.append(f"'{escaped_term}'")
            else:
                formatted_terms.append(str(term))
        
        terms_str = ", ".join(formatted_terms)
        return f"{self.predicate}({terms_str})"

@dataclass 
class PrologRule:
    """Rappresenta una regola Prolog nella forma head :- body."""
    head: str
    body: str
    
    def __str__(self):
        if not self.body:
            return f"{self.head}."
        return f"{self.head} :- {self.body}."
    
    def to_prolog_string(self):
        """Converte la regola in stringa Prolog valida."""
        if not self.body:
            return f"{self.head}."
        
        # Gestisce le virgolette sostituendole con atomi
        body_fixed = self.body.replace("'", "")
        return f"({self.head} :- {body_fixed})"

class PrologEngine:
    """
    Motore di inferenza Prolog per il sistema ricette.
    """
    
    def __init__(self):
        """Inizializza il motore Prolog."""
        self.logger = logging.getLogger(__name__)
        
        if not PYSWIP_AVAILABLE:
            raise ImportError("PySwip non disponibile. Installare con: pip install pyswip")
        
        try:
            self.prolog = Prolog()
            self.facts = []
            self.rules = []
            self.logger.info("Motore Prolog inizializzato con successo")
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione di Prolog: {e}")
            raise
    
    def add_fact(self, predicate: str, *terms):
        """
        Aggiunge un fatto al motore Prolog.
        
        Args:
            predicate: Nome del predicato
            *terms: Termini del predicato
        """
        try:
            fact = PrologFact(predicate, list(terms))
            self.facts.append(fact)
            
            # Aggiungi al motore Prolog
            prolog_string = fact.to_prolog_string()
            self.prolog.assertz(prolog_string)
            
            self.logger.debug(f"Fatto Prolog aggiunto: {prolog_string}")
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta del fatto {predicate}({terms}): {e}")
    
    def add_rule(self, head: str, body: str = ""):
        """
        Aggiunge una regola al motore Prolog.
        
        Args:
            head: Testa della regola
            body: Corpo della regola (opzionale)
        """
        try:
            rule = PrologRule(head, body)
            self.rules.append(rule)
            
            # Aggiungi al motore Prolog
            prolog_string = rule.to_prolog_string()
            self.prolog.assertz(prolog_string)
            
            self.logger.debug(f"Regola Prolog aggiunta: {prolog_string}")
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta della regola {head}: {e}")
    
    def query(self, query_string: str) -> List[Dict[str, Any]]:
        """
        Esegue una query Prolog.
        
        Args:
            query_string: Query Prolog da eseguire
            
        Returns:
            Lista di risultati con sostituzioni
        """
        try:
            self.logger.info(f"Eseguendo query Prolog: {query_string}")
            
            # Assicurati che la query non termini con un punto
            clean_query = query_string.rstrip('.')
            
            # Esegui la query
            results = list(self.prolog.query(clean_query))
            
            self.logger.info(f"Query completata. Risultati: {len(results)}")
            return results
            
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione della query '{query_string}': {e}")
            return []
    
    def query_all(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Trova tutte le soluzioni per un predicato.
        
        Args:
            predicate: Predicato da interrogare
            
        Returns:
            Lista di tutte le soluzioni
        """
        return self.query(predicate)
    
    def load_default_rules(self):
        """Carica le regole di default per il dominio delle ricette."""
        
        # Regole per ingredienti vegetali  
        self.add_rule(
            "vegetale(X)",
            "ingrediente(X), categoria(X, vegetale)"
        )
        
        self.add_rule(
            "vegetale(X)", 
            "ingrediente(X), categoria(X, frutta)"
        )
        
        # Regole per ricette vegetariane (usa i fatti is_vegetariano)
        self.add_rule(
            "ricetta_vegetariana(R)",
            "ricetta(R), is_vegetariano(R)"
        )
        
        # Regole per ricette light (< 200 calorie per ingrediente medio)
        self.add_rule(
            "ingrediente_light(I)",
            "ingrediente(I), calorie(I, Cal), Cal < 200"
        )
        
        # Regole per ricette veloci (< 30 minuti)
        self.add_rule(
            "ricetta_veloce(R)",
            "ricetta(R), tempo(R, T), T < 30"
        )
        
        # Regole per ingredienti di origine animale
        self.add_rule(
            "animale(X)",
            "ingrediente(X), categoria(X, carne)"
        )
        
        self.add_rule(
            "animale(X)",
            "ingrediente(X), categoria(X, latticini)"
        )
        
        # Regole per allergeni
        self.add_rule(
            "contiene_allergenici(R)",
            "ricetta(R), ha_ingrediente(R, I), allergenico(I)"
        )
        
        self.logger.info("Regole Prolog di default caricate")
        
        # Regole per compatibilità alimentare
        self.add_rule(
            "compatibile_dieta_vegana(R)",
            "ricetta(R), forall(ha_ingrediente(R, I), not(animale(I)))"
        )
        
        # Regole per ingredienti di origine animale
        self.add_rule(
            "animale(X)",
            "ingrediente(X), categoria(X, carne)"
        )
        
        self.add_rule(
            "animale(X)",
            "ingrediente(X), categoria(X, latticini)"
        )
        
        # Regole per calcolo calorie totali (semplificata)
        self.add_rule(
            "calorie_totali(R, Cal)",
            "ricetta(R), findall(C, (ha_ingrediente(R, I), calorie(I, C)), Lista), sum_list(Lista, Cal)"
        )
        
        # Regole per allergenici
        self.add_rule(
            "contiene_allergenici(R)",
            "ricetta(R), ha_ingrediente(R, I), allergenico(I)"
        )
        
        # === REGOLE AVANZATE BASATE SUI NUOVI PREDICATI ===
        
        # Regole per difficoltà
        self.add_rule(
            "ricetta_facile(R)",
            "ricetta(R), difficolta(R, facile)"
        )
        
        self.add_rule(
            "ricetta_difficile(R)",
            "ricetta(R), difficolta(R, difficile)"
        )
        
        # Regole per tipo piatto
        self.add_rule(
            "primi_piatti(R)",
            "ricetta(R), tipo_piatto(R, primo)"
        )
        
        self.add_rule(
            "secondi_piatti(R)",
            "ricetta(R), tipo_piatto(R, secondo)"
        )
        
        self.add_rule(
            "dolci(R)",
            "ricetta(R), tipo_piatto(R, dolce)"
        )
        
        # Regole per metodi di cottura
        self.add_rule(
            "ricette_al_forno(R)",
            "ricetta(R), metodo_cottura(R, al_forno)"
        )
        
        self.add_rule(
            "ricette_crude(R)",
            "ricetta(R), metodo_cottura(R, crudo)"
        )
        
        # Regole stagionali
        self.add_rule(
            "ricetta_estiva(R)",
            "ricetta(R), stagionalita(R, estate)"
        )
        
        self.add_rule(
            "ricetta_invernale(R)",
            "ricetta(R), stagionalita(R, autunno_inverno)"
        )
        
        # Regole per occasioni
        self.add_rule(
            "ricetta_quotidiana(R)",
            "ricetta(R), occasione_consumo(R, quotidiana)"
        )
        
        self.add_rule(
            "ricetta_festiva(R)",
            "ricetta(R), occasione_consumo(R, festiva)"
        )
        
        self.add_rule(
            "ricetta_elegante(R)",
            "ricetta(R), occasione_consumo(R, cena_elegante)"
        )
        
        # Regole per costi
        self.add_rule(
            "ricetta_economica(R)",
            "ricetta(R), costo_stimato(R, C), C < 10"
        )
        
        self.add_rule(
            "ricetta_costosa(R)",
            "ricetta(R), costo_stimato(R, C), C > 20"
        )
        
        # Regole per rating
        self.add_rule(
            "ricetta_ben_valutata(R)",
            "ricetta(R), rating(R, Rat), Rat > 4.5"
        )
        
        # Regole per popolarità
        self.add_rule(
            "ricetta_popolare(R)",
            "ricetta(R), numero_preparazioni(R, N), N > 10000"
        )
        
        # Regole per calorie
        self.add_rule(
            "ricetta_light_calorie(R)",
            "ricetta(R), calorie_porzione(R, Cal), Cal < 300"
        )
        
        self.add_rule(
            "ricetta_calorica(R)",
            "ricetta(R), calorie_porzione(R, Cal), Cal > 500"
        )
        
        # Regole per ingredienti
        self.add_rule(
            "ingrediente_locale(I)",
            "ingrediente(I), provenienza(I, locale)"
        )
        
        self.add_rule(
            "ingrediente_mediterraneo(I)",
            "ingrediente(I), provenienza(I, mediterraneo)"
        )
        
        self.add_rule(
            "ingrediente_biologico(I)",
            "ingrediente(I), metodo_produzione(I, biologico)"
        )
        
        self.add_rule(
            "ingrediente_crudo(I)",
            "ingrediente(I), livello_trasformazione(I, crudo)"
        )
        
        # Regole per valori nutrizionali
        self.add_rule(
            "ingrediente_basso_indice_glicemico(I)",
            "ingrediente(I), indice_glicemico(I, IG), IG < 55"
        )
        
        self.add_rule(
            "ingrediente_ricco_acqua(I)",
            "ingrediente(I), contenuto_acqua(I, A), A > 80"
        )
        
        # Regole combinate complesse
        self.add_rule(
            "ricetta_salutare(R)",
            "ricetta_vegetariana(R), ricetta_light_calorie(R), not(contiene_allergenici(R))"
        )
        
        self.add_rule(
            "ricetta_tradizionale_italiana(R)",
            "ricetta(R), tipo_cucina(R, italiana), numero_preparazioni(R, N), N > 5000"
        )
        
        self.add_rule(
            "ricetta_perfetta_per_cena(R)",
            "ricetta_elegante(R), ricetta_ben_valutata(R), tempo_totale(R, T), T < 90"
        )
        
        # Regole per combinazioni di ingredienti
        self.add_rule(
            "ricetta_con_ingredienti_locali(R)",
            "ricetta(R), forall(ha_ingrediente(R, I), ingrediente_locale(I))"
        )
        
        self.add_rule(
            "ricetta_biologica(R)",
            "ricetta(R), forall(ha_ingrediente(R, I), ingrediente_biologico(I))"
        )
        
        self.logger.info("Regole Prolog avanzate caricate")
    
    def import_from_kb(self, kb_manager):
        """
        Importa dati dalla Knowledge Base nel motore Prolog.
        
        Args:
            kb_manager: Istanza del RecipeKnowledgeBase
        """
        try:
            # Importa ingredienti
            ingredients = kb_manager.get_all_ingredients()
            for ingredient_id in ingredients:
                # Fatto base: ingrediente
                self.add_fact("ingrediente", ingredient_id)
            
            # Importa ricette
            recipes = kb_manager.get_all_recipes()
            for recipe_id in recipes:
                # Fatto base: ricetta
                self.add_fact("ricetta", recipe_id)
                
                # Importa relazioni ricetta-ingrediente
                recipe_ingredients = kb_manager.get_recipe_ingredients(recipe_id)
                for ingredient_id in recipe_ingredients:
                    self.add_fact("ha_ingrediente", recipe_id, ingredient_id)
            
            # Importa proprietà tramite SPARQL
            self._import_properties_from_kb(kb_manager)
            
            self.logger.info(f"Importati {len(ingredients)} ingredienti e {len(recipes)} ricette in Prolog")
            
        except Exception as e:
            self.logger.error(f"Errore nell'importazione dalla KB: {e}")
    
    def _import_properties_from_kb(self, kb_manager):
        """Importa proprietà specifiche dai dati strutturati."""
        try:
            # Invece di usare SPARQL (che potrebbe essere vuoto), 
            # importiamo dai dati che abbiamo appena caricato
            from src.data_loader import DataLoader
            data_loader = DataLoader('data')
            kb_data = data_loader.generate_initial_kb()
            
            # === IMPORTA TUTTE LE PROPRIETÀ DEGLI INGREDIENTI ===
            for ingredient_id, ingredient_data in kb_data['ingredienti'].items():
                # Categoria alimentare
                if 'categoriaAlimentare' in ingredient_data:
                    category = ingredient_data['categoriaAlimentare'].lower()
                    self.add_fact("categoria", ingredient_id, category)
                
                # Colore
                if 'colore' in ingredient_data:
                    self.add_fact("colore", ingredient_id, ingredient_data['colore'])
                
                # Consistenza
                if 'consistenza' in ingredient_data:
                    self.add_fact("consistenza", ingredient_id, ingredient_data['consistenza'])
                
                # Origine
                if 'origine' in ingredient_data:
                    self.add_fact("origine", ingredient_id, ingredient_data['origine'])
                
                # Stagionalità
                if 'stagionalita' in ingredient_data:
                    self.add_fact("stagionalita_ingrediente", ingredient_id, ingredient_data['stagionalita'])
                
                # Provenienza geografica
                if 'provenienza_geografica' in ingredient_data:
                    self.add_fact("provenienza", ingredient_id, ingredient_data['provenienza_geografica'])
                
                # Metodo di produzione
                if 'metodo_produzione' in ingredient_data:
                    self.add_fact("metodo_produzione", ingredient_id, ingredient_data['metodo_produzione'])
                
                # Livello di trasformazione
                if 'livello_trasformazione' in ingredient_data:
                    self.add_fact("livello_trasformazione", ingredient_id, ingredient_data['livello_trasformazione'])
                
                # Valori numerici
                if 'peso_medio_g' in ingredient_data:
                    self.add_fact("peso_medio", ingredient_id, float(ingredient_data['peso_medio_g']))
                
                if 'durata_conservazione_giorni' in ingredient_data:
                    self.add_fact("durata_conservazione", ingredient_id, float(ingredient_data['durata_conservazione_giorni']))
                
                if 'prezzo_per_kg' in ingredient_data:
                    self.add_fact("prezzo_kg", ingredient_id, float(ingredient_data['prezzo_per_kg']))
                
                if 'contenuto_acqua_perc' in ingredient_data:
                    self.add_fact("contenuto_acqua", ingredient_id, float(ingredient_data['contenuto_acqua_perc']))
                
                if 'densita_nutrizionale_kcal100g' in ingredient_data:
                    calories = float(ingredient_data['densita_nutrizionale_kcal100g'])
                    self.add_fact("calorie", ingredient_id, calories)
                
                if 'indice_glicemico' in ingredient_data:
                    self.add_fact("indice_glicemico", ingredient_id, float(ingredient_data['indice_glicemico']))
                
                # Allergeni (per ora basato su euristica)
                if ingredient_data.get('isAlergenico', False):
                    self.add_fact("allergenico", ingredient_id)
            
            # === IMPORTA TUTTE LE PROPRIETÀ DELLE RICETTE ===
            for recipe_id, recipe_data in kb_data['ricette'].items():
                # Tipo cucina
                if 'tipo_cucina' in recipe_data:
                    self.add_fact("tipo_cucina", recipe_id, recipe_data['tipo_cucina'])
                
                # Difficoltà
                if 'difficolta' in recipe_data:
                    self.add_fact("difficolta", recipe_id, recipe_data['difficolta'])
                
                # Tipo piatto
                if 'tipo_piatto' in recipe_data:
                    self.add_fact("tipo_piatto", recipe_id, recipe_data['tipo_piatto'])
                
                # Metodo cottura
                if 'metodo_cottura' in recipe_data:
                    self.add_fact("metodo_cottura", recipe_id, recipe_data['metodo_cottura'])
                
                # Stagionalità
                if 'stagionalita' in recipe_data:
                    self.add_fact("stagionalita", recipe_id, recipe_data['stagionalita'])
                
                # Dieta speciale
                if 'dieta_speciale' in recipe_data:
                    dieta = recipe_data['dieta_speciale']
                    self.add_fact("dieta_speciale", recipe_id, dieta)
                    
                    # Deriva proprietà vegetariane
                    if dieta.lower() in ['vegetariana', 'vegana']:
                        self.add_fact("is_vegetariano", recipe_id)
                
                # Occasione consumo
                if 'occasione_consumo' in recipe_data:
                    self.add_fact("occasione_consumo", recipe_id, recipe_data['occasione_consumo'])
                
                # Valori numerici ricette
                if 'numero_ingredienti' in recipe_data:
                    self.add_fact("numero_ingredienti", recipe_id, int(recipe_data['numero_ingredienti']))
                
                if 'numero_porzioni' in recipe_data:
                    self.add_fact("numero_porzioni", recipe_id, int(recipe_data['numero_porzioni']))
                
                if 'tempo_preparazione_min' in recipe_data:
                    tempo = int(recipe_data['tempo_preparazione_min'])
                    self.add_fact("tempo_preparazione", recipe_id, tempo)
                    self.add_fact("tempo", recipe_id, tempo)  # Alias per compatibilità
                
                if 'tempo_cottura_min' in recipe_data:
                    self.add_fact("tempo_cottura", recipe_id, int(recipe_data['tempo_cottura_min']))
                
                if 'tempo_totale_min' in recipe_data:
                    self.add_fact("tempo_totale", recipe_id, int(recipe_data['tempo_totale_min']))
                
                if 'costo_stimato_euro' in recipe_data:
                    self.add_fact("costo_stimato", recipe_id, float(recipe_data['costo_stimato_euro']))
                
                if 'rating_medio' in recipe_data:
                    self.add_fact("rating", recipe_id, float(recipe_data['rating_medio']))
                
                if 'numero_preparazioni' in recipe_data:
                    self.add_fact("numero_preparazioni", recipe_id, int(recipe_data['numero_preparazioni']))
                
                if 'calorie_per_porzione' in recipe_data:
                    self.add_fact("calorie_porzione", recipe_id, int(recipe_data['calorie_per_porzione']))
                
            self.logger.info("Tutte le proprietà dai dati strutturati importate in Prolog")
            
        except Exception as e:
            self.logger.error(f"Errore nell'importazione proprietà: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Restituisce statistiche del motore Prolog."""
        return {
            'fatti': len(self.facts),
            'regole': len(self.rules),
            'predicati_unici': len(set(fact.predicate for fact in self.facts))
        }
    
    def reset(self):
        """Resetta il motore Prolog rimuovendo tutti i fatti e le regole."""
        try:
            # Retract all facts and rules
            self.prolog.retractall("_")
            self.facts.clear()
            self.rules.clear()
            self.logger.info("Motore Prolog resettato")
        except Exception as e:
            self.logger.error(f"Errore nel reset del motore Prolog: {e}")
    
    def explain_query(self, query_string: str) -> str:
        """
        Fornisce una spiegazione di come una query viene risolta.
        
        Args:
            query_string: Query da spiegare
            
        Returns:
            Spiegazione testuale del processo di risoluzione
        """
        # Implementazione semplificata - in un sistema reale useresti trace/0
        explanation = f"Query: {query_string}\n"
        explanation += "Processo di risoluzione:\n"
        
        results = self.query(query_string)
        if results:
            explanation += f"✅ Trovate {len(results)} soluzioni:\n"
            for i, result in enumerate(results, 1):
                explanation += f"  {i}. {result}\n"
        else:
            explanation += "❌ Nessuna soluzione trovata\n"
        
        return explanation

class PrologQueryBuilder:
    """Helper per costruire query Prolog complesse."""
    
    # === QUERY BASE ===
    @staticmethod
    def find_vegetarian_recipes():
        """Query per trovare ricette vegetariane."""
        return "ricetta_vegetariana(R)"
    
    @staticmethod
    def find_recipes_with_ingredient(ingredient):
        """Query per trovare ricette che contengono un ingrediente specifico."""
        return f"ha_ingrediente(R, '{ingredient}')"
    
    @staticmethod
    def find_light_recipes():
        """Query per trovare ricette light."""
        return "ricetta_light_calorie(R)"
    
    @staticmethod
    def find_quick_recipes(max_time=30):
        """Query per trovare ricette veloci."""
        return f"ricetta(R), tempo(R, T), T < {max_time}"
    
    @staticmethod
    def find_recipes_by_calories(max_calories):
        """Query per trovare ricette sotto una certa soglia calorica."""
        return f"ricetta(R), calorie_porzione(R, Cal), Cal < {max_calories}"
    
    # === QUERY PER TIPO CUCINA ===
    @staticmethod
    def find_italian_recipes():
        """Query per trovare ricette italiane."""
        return "ricetta(R), tipo_cucina(R, italiana)"
    
    @staticmethod
    def find_recipes_by_cuisine(cuisine_type):
        """Query per trovare ricette per tipo di cucina."""
        return f"ricetta(R), tipo_cucina(R, {cuisine_type})"
    
    # === QUERY PER DIFFICOLTÀ ===
    @staticmethod
    def find_easy_recipes():
        """Query per trovare ricette facili."""
        return "ricetta_facile(R)"
    
    @staticmethod
    def find_difficult_recipes():
        """Query per trovare ricette difficili."""
        return "ricetta_difficile(R)"
    
    # === QUERY PER TIPO PIATTO ===
    @staticmethod
    def find_first_courses():
        """Query per trovare primi piatti."""
        return "primi_piatti(R)"
    
    @staticmethod
    def find_second_courses():
        """Query per trovare secondi piatti."""
        return "secondi_piatti(R)"
    
    @staticmethod
    def find_desserts():
        """Query per trovare dolci."""
        return "dolci(R)"
    
    # === QUERY PER METODO COTTURA ===
    @staticmethod
    def find_baked_recipes():
        """Query per trovare ricette al forno."""
        return "ricette_al_forno(R)"
    
    @staticmethod
    def find_raw_recipes():
        """Query per trovare ricette crude."""
        return "ricette_crude(R)"
    
    # === QUERY STAGIONALI ===
    @staticmethod
    def find_summer_recipes():
        """Query per trovare ricette estive."""
        return "ricetta_estiva(R)"
    
    @staticmethod
    def find_winter_recipes():
        """Query per trovare ricette invernali."""
        return "ricetta_invernale(R)"
    
    # === QUERY PER OCCASIONI ===
    @staticmethod
    def find_daily_recipes():
        """Query per trovare ricette quotidiane."""
        return "ricetta_quotidiana(R)"
    
    @staticmethod
    def find_festive_recipes():
        """Query per trovare ricette festive."""
        return "ricetta_festiva(R)"
    
    @staticmethod
    def find_elegant_recipes():
        """Query per trovare ricette eleganti."""
        return "ricetta_elegante(R)"
    
    # === QUERY PER COSTI ===
    @staticmethod
    def find_cheap_recipes():
        """Query per trovare ricette economiche."""
        return "ricetta_economica(R)"
    
    @staticmethod
    def find_expensive_recipes():
        """Query per trovare ricette costose."""
        return "ricetta_costosa(R)"
    
    # === QUERY PER RATING E POPOLARITÀ ===
    @staticmethod
    def find_highly_rated_recipes():
        """Query per trovare ricette ben valutate."""
        return "ricetta_ben_valutata(R)"
    
    @staticmethod
    def find_popular_recipes():
        """Query per trovare ricette popolari."""
        return "ricetta_popolare(R)"
    
    # === QUERY PER INGREDIENTI ===
    @staticmethod
    def find_local_ingredients():
        """Query per trovare ingredienti locali."""
        return "ingrediente_locale(I)"
    
    @staticmethod
    def find_mediterranean_ingredients():
        """Query per trovare ingredienti mediterranei."""
        return "ingrediente_mediterraneo(I)"
    
    @staticmethod
    def find_organic_ingredients():
        """Query per trovare ingredienti biologici."""
        return "ingrediente_biologico(I)"
    
    @staticmethod
    def find_low_glycemic_ingredients():
        """Query per trovare ingredienti a basso indice glicemico."""
        return "ingrediente_basso_indice_glicemico(I)"
    
    # === QUERY COMPLESSE ===
    @staticmethod
    def find_healthy_recipes():
        """Query per trovare ricette salutari."""
        return "ricetta_salutare(R)"
    
    @staticmethod
    def find_traditional_italian_recipes():
        """Query per trovare ricette tradizionali italiane."""
        return "ricetta_tradizionale_italiana(R)"
    
    @staticmethod
    def find_perfect_dinner_recipes():
        """Query per trovare ricette perfette per cena."""
        return "ricetta_perfetta_per_cena(R)"
    
    @staticmethod
    def find_recipes_with_local_ingredients():
        """Query per trovare ricette con ingredienti locali."""
        return "ricetta_con_ingredienti_locali(R)"
    
    @staticmethod
    def find_organic_recipes():
        """Query per trovare ricette biologiche."""
        return "ricetta_biologica(R)"
    
    @staticmethod
    def complex_query_example():
        """Esempio di query complessa che combina più criteri."""
        return """
        ricetta_vegetariana(R), 
        ricetta_veloce(R), 
        not(contiene_allergenici(R))
        """
    
    # === QUERY PERSONALIZZATE AVANZATE ===
    @staticmethod
    def find_recipes_for_diabetics():
        """Query per trovare ricette adatte ai diabetici."""
        return """
        ricetta(R),
        forall(ha_ingrediente(R, I), 
               ingrediente_basso_indice_glicemico(I))
        """
    
    @staticmethod
    def find_quick_and_healthy():
        """Query per ricette veloci e salutari."""
        return """
        ricetta_veloce(R),
        ricetta_light_calorie(R),
        not(contiene_allergenici(R))
        """
    
    @staticmethod
    def find_budget_family_recipes():
        """Query per ricette economiche per famiglie."""
        return """
        ricetta_economica(R),
        numero_porzioni(R, P), P >= 4,
        ricetta_facile(R)
        """
    
    @staticmethod
    def find_gourmet_recipes():
        """Query per ricette gourmet."""
        return """
        ricetta_elegante(R),
        ricetta_ben_valutata(R),
        difficolta(R, difficile)
        """
    
    @staticmethod
    def find_seasonal_local_recipes(season):
        """Query per ricette stagionali con ingredienti locali."""
        return f"""
        ricetta(R),
        stagionalita(R, {season}),
        forall(ha_ingrediente(R, I), ingrediente_locale(I))
        """
