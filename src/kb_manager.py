"""
Knowledge Base Manager per la gestione di Knowledge Graphs e Ontologie.

Questo modulo implementa la gestione della Knowledge Base utilizzando RDF/OWL,
con supporto per la definizione di ontologie, popolamento della KB e query SPARQL.
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef, BNode
from rdflib.namespace import XSD
from typing import Dict, List, Tuple, Any, Optional
import json
import logging

# Definizione dei namespace per l'ontologia delle ricette
RECIPE = Namespace("http://example.org/recipe#")
ONTO = Namespace("http://example.org/ontology#")

class RecipeKnowledgeBase:
    """
    Gestisce la Knowledge Base per le ricette culinarie.
    Implementa un'ontologia strutturata con classi, proprietà e individui.
    """
    
    def __init__(self):
        """Inizializza la Knowledge Base con l'ontologia di base."""
        self.graph = Graph()
        self.graph.bind("recipe", RECIPE)
        self.graph.bind("onto", ONTO)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdfs", RDFS)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Inizializza l'ontologia di base
        self._init_ontology()
    
    def _init_ontology(self):
        """
        Definisce l'ontologia di base per le ricette culinarie.
        Include classi, proprietà, domini, range e gerarchie.
        """
        # === DEFINIZIONE DELLE CLASSI ===
        
        # Classi principali
        self.graph.add((ONTO.Ricetta, RDF.type, OWL.Class))
        self.graph.add((ONTO.Ingrediente, RDF.type, OWL.Class))
        self.graph.add((ONTO.Utensile, RDF.type, OWL.Class))
        self.graph.add((ONTO.CategoriaAlimentare, RDF.type, OWL.Class))
        self.graph.add((ONTO.TipoCucina, RDF.type, OWL.Class))
        
        # Sottoclassi di CategoriaAlimentare
        self.graph.add((ONTO.Vegetale, RDF.type, OWL.Class))
        self.graph.add((ONTO.Vegetale, RDFS.subClassOf, ONTO.CategoriaAlimentare))
        
        self.graph.add((ONTO.Frutta, RDF.type, OWL.Class))
        self.graph.add((ONTO.Frutta, RDFS.subClassOf, ONTO.CategoriaAlimentare))
        
        self.graph.add((ONTO.Carne, RDF.type, OWL.Class))
        self.graph.add((ONTO.Carne, RDFS.subClassOf, ONTO.CategoriaAlimentare))
        
        self.graph.add((ONTO.Latticini, RDF.type, OWL.Class))
        self.graph.add((ONTO.Latticini, RDFS.subClassOf, ONTO.CategoriaAlimentare))
        
        self.graph.add((ONTO.Cereali, RDF.type, OWL.Class))
        self.graph.add((ONTO.Cereali, RDFS.subClassOf, ONTO.CategoriaAlimentare))
        
        # === DEFINIZIONE DELLE PROPRIETÀ ===
        
        # Proprietà oggetto (Object Properties)
        self.graph.add((ONTO.haIngrediente, RDF.type, OWL.ObjectProperty))
        self.graph.add((ONTO.haIngrediente, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.haIngrediente, RDFS.range, ONTO.Ingrediente))
        
        self.graph.add((ONTO.utilizzaUtensile, RDF.type, OWL.ObjectProperty))
        self.graph.add((ONTO.utilizzaUtensile, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.utilizzaUtensile, RDFS.range, ONTO.Utensile))
        
        self.graph.add((ONTO.appartieneCategoriaAlimentare, RDF.type, OWL.ObjectProperty))
        self.graph.add((ONTO.appartieneCategoriaAlimentare, RDFS.domain, ONTO.Ingrediente))
        self.graph.add((ONTO.appartieneCategoriaAlimentare, RDFS.range, ONTO.CategoriaAlimentare))
        
        self.graph.add((ONTO.haTipoCucina, RDF.type, OWL.ObjectProperty))
        self.graph.add((ONTO.haTipoCucina, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.haTipoCucina, RDFS.range, ONTO.TipoCucina))
        
        # Proprietà dati (Data Properties) - Funzionali
        self.graph.add((ONTO.tempoDiPreparazione, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.tempoDiPreparazione, RDF.type, OWL.FunctionalProperty))
        self.graph.add((ONTO.tempoDiPreparazione, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.tempoDiPreparazione, RDFS.range, XSD.integer))
        
        self.graph.add((ONTO.caloriePerPorzione, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.caloriePerPorzione, RDF.type, OWL.FunctionalProperty))
        self.graph.add((ONTO.caloriePerPorzione, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.caloriePerPorzione, RDFS.range, XSD.integer))
        
        self.graph.add((ONTO.caloriePer100g, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.caloriePer100g, RDF.type, OWL.FunctionalProperty))
        self.graph.add((ONTO.caloriePer100g, RDFS.domain, ONTO.Ingrediente))
        self.graph.add((ONTO.caloriePer100g, RDFS.range, XSD.integer))
        
        self.graph.add((ONTO.porzioni, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.porzioni, RDF.type, OWL.FunctionalProperty))
        self.graph.add((ONTO.porzioni, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.porzioni, RDFS.range, XSD.integer))
        
        self.graph.add((ONTO.istruzioni, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.istruzioni, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.istruzioni, RDFS.range, XSD.string))
        
        # Proprietà booleane
        self.graph.add((ONTO.isVegetariano, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.isVegetariano, RDFS.domain, ONTO.Ricetta))
        self.graph.add((ONTO.isVegetariano, RDFS.range, XSD.boolean))
        
        self.graph.add((ONTO.isAlergenico, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.isAlergenico, RDFS.domain, ONTO.Ingrediente))
        self.graph.add((ONTO.isAlergenico, RDFS.range, XSD.boolean))
        
        # Proprietà gerarchiche per nutrienti
        self.graph.add((ONTO.haNutriente, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.haNutriente, RDFS.domain, ONTO.Ingrediente))
        self.graph.add((ONTO.haNutriente, RDFS.range, XSD.string))
        
        self.graph.add((ONTO.haProteine, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.haProteine, RDFS.subPropertyOf, ONTO.haNutriente))
        self.graph.add((ONTO.haProteine, RDFS.domain, ONTO.Ingrediente))
        self.graph.add((ONTO.haProteine, RDFS.range, XSD.float))
        
        self.graph.add((ONTO.haCarboidrati, RDF.type, OWL.DatatypeProperty))
        self.graph.add((ONTO.haCarboidrati, RDFS.subPropertyOf, ONTO.haNutriente))
        self.graph.add((ONTO.haCarboidrati, RDFS.domain, ONTO.Ingrediente))
        self.graph.add((ONTO.haCarboidrati, RDFS.range, XSD.float))
        
        # Log completamento inizializzazione ontologia
        logger = logging.getLogger(__name__)
        logger.info("Ontologia di base inizializzata con successo")
    
    def add_recipe(self, recipe_id: str, recipe_data: Dict[str, Any]) -> bool:
        """
        Aggiunge una ricetta alla Knowledge Base.
        
        Args:
            recipe_id: Identificatore univoco della ricetta
            recipe_data: Dizionario con i dati della ricetta
            
        Returns:
            True se l'aggiunta è avvenuta con successo, False altrimenti
        """
        try:
            recipe_uri = RECIPE[recipe_id]
            
            # Aggiunge la ricetta come istanza della classe Ricetta
            self.graph.add((recipe_uri, RDF.type, ONTO.Ricetta))
            
            # Aggiunge le proprietà della ricetta
            if 'nome' in recipe_data:
                self.graph.add((recipe_uri, RDFS.label, Literal(recipe_data['nome'])))
            
            if 'tempoDiPreparazione' in recipe_data:
                self.graph.add((recipe_uri, ONTO.tempoDiPreparazione, 
                              Literal(recipe_data['tempoDiPreparazione'], datatype=XSD.integer)))
            
            if 'caloriePerPorzione' in recipe_data:
                self.graph.add((recipe_uri, ONTO.caloriePerPorzione, 
                              Literal(recipe_data['caloriePerPorzione'], datatype=XSD.integer)))
            
            if 'porzioni' in recipe_data:
                self.graph.add((recipe_uri, ONTO.porzioni, 
                              Literal(recipe_data['porzioni'], datatype=XSD.integer)))
            
            if 'istruzioni' in recipe_data:
                self.graph.add((recipe_uri, ONTO.istruzioni, 
                              Literal(recipe_data['istruzioni'])))
            
            if 'isVegetariano' in recipe_data:
                self.graph.add((recipe_uri, ONTO.isVegetariano, 
                              Literal(recipe_data['isVegetariano'], datatype=XSD.boolean)))
            
            # Aggiunge ingredienti
            if 'ingredienti' in recipe_data:
                for ingrediente_id in recipe_data['ingredienti']:
                    ingrediente_uri = RECIPE[ingrediente_id]
                    self.graph.add((recipe_uri, ONTO.haIngrediente, ingrediente_uri))
            
            # Aggiunge utensili
            if 'utensili' in recipe_data:
                for utensile_id in recipe_data['utensili']:
                    utensile_uri = RECIPE[utensile_id]
                    self.graph.add((recipe_uri, ONTO.utilizzaUtensile, utensile_uri))
            
            # Aggiunge tipo di cucina
            if 'tipoCucina' in recipe_data:
                tipo_cucina_uri = RECIPE[recipe_data['tipoCucina']]
                self.graph.add((recipe_uri, ONTO.haTipoCucina, tipo_cucina_uri))
            
            self.logger.info(f"Ricetta {recipe_id} aggiunta con successo")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta della ricetta {recipe_id}: {e}")
            return False
    
    def add_ingredient(self, ingredient_id: str, ingredient_data: Dict[str, Any]) -> bool:
        """
        Aggiunge un ingrediente alla Knowledge Base e lo salva nel CSV.
        
        Args:
            ingredient_id: Identificatore univoco dell'ingrediente
            ingredient_data: Dizionario con i dati dell'ingrediente
            
        Returns:
            True se l'aggiunta è avvenuta con successo, False altrimenti
        """
        try:
            ingredient_uri = RECIPE[ingredient_id]
            
            # Aggiunge l'ingrediente come istanza della classe Ingrediente
            self.graph.add((ingredient_uri, RDF.type, ONTO.Ingrediente))
            
            # Aggiunge le proprietà dell'ingrediente
            if 'nome' in ingredient_data:
                self.graph.add((ingredient_uri, RDFS.label, 
                              Literal(ingredient_data['nome'])))
            
            if 'caloriePer100g' in ingredient_data:
                self.graph.add((ingredient_uri, ONTO.caloriePer100g, 
                              Literal(ingredient_data['caloriePer100g'], datatype=XSD.integer)))
            
            if 'isAlergenico' in ingredient_data:
                self.graph.add((ingredient_uri, ONTO.isAlergenico, 
                              Literal(ingredient_data['isAlergenico'], datatype=XSD.boolean)))
            
            if 'proteine' in ingredient_data:
                self.graph.add((ingredient_uri, ONTO.haProteine, 
                              Literal(ingredient_data['proteine'], datatype=XSD.float)))
            
            if 'carboidrati' in ingredient_data:
                self.graph.add((ingredient_uri, ONTO.haCarboidrati, 
                              Literal(ingredient_data['carboidrati'], datatype=XSD.float)))
            
            # Aggiunge categoria alimentare
            if 'categoriaAlimentare' in ingredient_data:
                categoria_uri = ONTO[ingredient_data['categoriaAlimentare']]
                self.graph.add((ingredient_uri, ONTO.appartieneCategoriaAlimentare, categoria_uri))
            
            self.logger.info(f"Ingrediente {ingredient_id} aggiunto con successo alla KB")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta dell'ingrediente {ingredient_id}: {e}")
            return False
    
    def load_initial_kb(self, kb_file_path: str) -> bool:
        """
        Carica la Knowledge Base iniziale da un file JSON.
        AGGIORNATO: Ora gestisce chiavi basate sui nomi reali degli ingredienti e ricette.
        
        Args:
            kb_file_path: Percorso del file JSON contenente la KB iniziale
            
        Returns:
            True se il caricamento è avvenuto con successo, False altrimenti
        """
        try:
            with open(kb_file_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            # Carica ingredienti (ora le chiavi sono i nomi reali)
            if 'ingredienti' in kb_data:
                for ingredient_name, ingredient_data in kb_data['ingredienti'].items():
                    # Crea un ID sicuro per URI RDF dal nome dell'ingrediente
                    safe_id = self._create_safe_uri_id(ingredient_name)
                    self.add_ingredient(safe_id, ingredient_data)
            
            # Carica ricette (ora le chiavi sono i nomi reali)
            if 'ricette' in kb_data:
                for recipe_name, recipe_data in kb_data['ricette'].items():
                    # Crea un ID sicuro per URI RDF dal nome della ricetta
                    safe_id = self._create_safe_uri_id(recipe_name)
                    
                    # Aggiorna gli ingredienti nelle ricette per usare gli ID sicuri
                    if 'ingredienti' in recipe_data:
                        safe_ingredients = []
                        for ing_name in recipe_data['ingredienti']:
                            safe_ingredients.append(self._create_safe_uri_id(ing_name))
                        recipe_data['ingredienti'] = safe_ingredients
                    
                    self.add_recipe(safe_id, recipe_data)

            self.logger.info(f"Knowledge Base caricata da {kb_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel caricamento della KB da {kb_file_path}: {e}")
            return False
    
    def _create_safe_uri_id(self, name: str) -> str:
        """
        Crea un ID sicuro per URI RDF a partire da un nome.
        Rimuove caratteri speciali e spazi, sostituendoli con underscore.
        
        Args:
            name: Nome originale da convertire
            
        Returns:
            ID sicuro per URI
        """
        import re
        # Rimuovi caratteri speciali e sostituisci spazi con underscore
        safe_id = re.sub(r'[^\w\s-]', '', name)  # Rimuovi caratteri speciali
        safe_id = re.sub(r'[-\s]+', '_', safe_id)  # Sostituisci spazi/trattini con underscore
        return safe_id.strip('_')  # Rimuovi underscore all'inizio/fine
    
    def query_sparql(self, query: str) -> List[Dict[str, Any]]:
        """
        Esegue una query SPARQL sulla Knowledge Base.
        
        Args:
            query: Query SPARQL da eseguire
            
        Returns:
            Lista di risultati della query
        """
        try:
            results = self.graph.query(query)
            return [dict(row.asdict()) for row in results]
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione della query SPARQL: {e}")
            return []
    
    def get_all_recipes(self) -> List[str]:
        """
        Restituisce tutti gli ID delle ricette nella KB.
        
        Returns:
            Lista degli ID delle ricette
        """
        query = """
        SELECT ?recipe WHERE {
            ?recipe rdf:type onto:Ricetta .
        }
        """
        results = self.query_sparql(query)
        return [str(result['recipe']).split('#')[-1] for result in results]
    
    def get_all_ingredients(self) -> List[str]:
        """
        Restituisce tutti gli ID degli ingredienti nella KB.
        
        Returns:
            Lista degli ID degli ingredienti
        """
        query = """
        SELECT ?ingredient WHERE {
            ?ingredient rdf:type onto:Ingrediente .
        }
        """
        results = self.query_sparql(query)
        return [str(result['ingredient']).split('#')[-1] for result in results]
    
    def get_recipe_ingredients(self, recipe_id: str) -> List[str]:
        """
        Restituisce gli ingredienti di una ricetta specifica.
        
        Args:
            recipe_id: ID della ricetta
            
        Returns:
            Lista degli ID degli ingredienti
        """
        query = f"""
        SELECT ?ingredient WHERE {{
            recipe:{recipe_id} onto:haIngrediente ?ingredient .
        }}
        """
        results = self.query_sparql(query)
        return [str(result['ingredient']).split('#')[-1] for result in results]
    
    def export_to_file(self, file_path: str, format: str = "turtle") -> bool:
        """
        Esporta la Knowledge Base in un file.
        
        Args:
            file_path: Percorso del file di output
            format: Formato di output (turtle, xml, nt, json-ld)
            
        Returns:
            True se l'esportazione è avvenuta con successo, False altrimenti
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.graph.serialize(format=format))
            self.logger.info(f"Knowledge Base esportata in {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Errore nell'esportazione: {e}")
            return False
    
    def get_kb_stats(self) -> Dict[str, int]:
        """
        Restituisce statistiche sulla Knowledge Base.
        
        Returns:
            Dizionario con le statistiche
        """
        stats = {
            'total_triples': len(self.graph),
            'ricette': len(self.get_all_recipes()),
            'ingredienti': len(self.get_all_ingredients())
        }
        
        # Conta le classi
        classes_query = "SELECT (COUNT(DISTINCT ?class) AS ?count) WHERE { ?class rdf:type owl:Class . }"
        result = self.query_sparql(classes_query)
        if result:
            stats['classi'] = int(result[0]['count'])
        
        # Conta le proprietà
        properties_query = """
        SELECT (COUNT(DISTINCT ?prop) AS ?count) WHERE { 
            { ?prop rdf:type owl:ObjectProperty . } 
            UNION 
            { ?prop rdf:type owl:DatatypeProperty . } 
        }
        """
        result = self.query_sparql(properties_query)
        if result:
            stats['proprieta'] = int(result[0]['count'])
        
        return stats
