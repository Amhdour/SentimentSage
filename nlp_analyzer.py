import spacy
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import numpy as np
from collections import Counter

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    """
    Extract named entities from text using spaCy.
    Returns a dictionary of entities grouped by type.
    """
    try:
        doc = nlp(text)
        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        # Count frequencies for each entity type
        for entity_type in entities:
            entity_counts = Counter(entities[entity_type])
            entities[entity_type] = [
                {"text": text, "count": count}
                for text, count in entity_counts.most_common()
            ]

        return entities
    except Exception as e:
        print(f"Error in entity extraction: {str(e)}")
        return {}

def get_key_phrases(text):
    """
    Extract key phrases using spaCy's noun chunks and dependency parsing.
    """
    try:
        doc = nlp(text)
        phrases = []

        # Extract noun chunks (phrases)
        for chunk in doc.noun_chunks:
            phrases.append({
                'text': chunk.text,
                'root': chunk.root.text,
                'type': 'noun_chunk'
            })

        # Extract verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                phrase = token.text
                # Get related objects
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj']:
                        phrase = f"{phrase} {child.text}"
                phrases.append({
                    'text': phrase,
                    'root': token.text,
                    'type': 'verb_phrase'
                })

        return phrases
    except Exception as e:
        print(f"Error in key phrase extraction: {str(e)}")
        return []

def perform_topic_modeling(texts, num_topics=3):
    """
    Perform topic modeling on a list of texts.
    """
    try:
        # Tokenize and preprocess texts
        processed_texts = [simple_preprocess(text) for text in texts]

        # Create dictionary
        dictionary = corpora.Dictionary(processed_texts)

        # Create corpus
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10
        )

        # Extract topics
        topics = []
        for topic_id in range(num_topics):
            topic_words = lda_model.show_topic(topic_id, 5)  # Get top 5 words
            topics.append({
                'id': topic_id,
                'words': [{'word': word, 'weight': float(weight)} 
                         for word, weight in topic_words]
            })

        return topics
    except Exception as e:
        print(f"Error in topic modeling: {str(e)}")
        return []

def get_nlp_analysis(text):
    """
    Perform comprehensive NLP analysis on the text.
    """
    result = {
        'entities': extract_entities(text),
        'key_phrases': get_key_phrases(text)
    }

    # Only perform topic modeling if text is long enough
    if len(text.split()) >= 20:
        result['topics'] = perform_topic_modeling([text])

    return result