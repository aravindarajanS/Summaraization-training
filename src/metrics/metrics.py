
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import sentence_bleu
import torch
from transformers import AutoTokenizer,BertForMaskedLM
import spacy
from gensim.models import KeyedVectors
from sent2vec.vectorizer import Vectorizer
from scipy.spatial.distance import cosine
import readability
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge

class Summarizer_evaluation:
    """
    A class for evaluating the quality of a text summary.
    """

    def __init__(self):

        """
            Download crawl word_embeddings
        """

        # Load word embeddings
        self.word_embeddings = KeyedVectors.load_word2vec_format('crawl-300d-2M-subword.vec', binary=False)

        self.porter_stemmer  = PorterStemmer()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################

    ### Accuracy ##########

    #######################


    def calculate_semantic_similarity(self, summary: str, document: str) -> float:
        """
        Calculates the semantic similarity between a summary and a document.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
            float: The semantic similarity score.
        """
        summary_words = summary.split(" ")
        document_words = document.split(" ")

        # Get word embeddings for summary and document words
        summary_embeddings = np.array([
            self.word_embeddings[word]
            for word in summary_words
            if word in self.word_embeddings
        ])
        document_embeddings = np.array([
            self.word_embeddings[word]
            for word in document_words
            if word in self.word_embeddings
        ])

        if len(summary_embeddings) == 0 or len(document_embeddings) == 0:
            return 0.0  # Handle case where no matching words found

        # Calculate the mean cosine similarity between summary and document embeddings
        similarity = np.mean(np.dot(summary_embeddings, document_embeddings.T))
        return similarity



    def lexical_overlap(self, summary: str, document: str) -> float:
        """
        Calculates the lexical overlap between a summary and a document.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
            float: The lexical overlap score.
        """
        summary_words = set(summary.split(" "))
        document_words = set(document.split(" "))

        # Calculate the lexical overlap as the ratio of overlapping words to the union of words
        overlap = len(summary_words & document_words) / len(summary_words | document_words)



        return overlap


    def calculate_rouge_scores(self, summary: str, document: str) -> float:

        """
        Calculates the rouge relevance score using rouge module.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
           float: Average of rouge-1, rouge-2 and rouge - l f1 score.
        """

        rouge = Rouge()

        scores = rouge.get_scores(document, summary)[0]
        rouge_1_f_measure = scores['rouge-1']['f']
        rouge_2_f_measure = scores['rouge-2']['f']
        rouge_l_f_measure = scores['rouge-l']['f']

        return (rouge_1_f_measure + rouge_2_f_measure + rouge_l_f_measure)/3


    def evaluate_accuracy(self, summary: str, document: str) -> float:
        """
        Evaluates the accuracy of a summary based on semantic similarity, \
             and lexical overlap.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
            float: The accuracy score.
        """
        semantic_similarity_score = self.calculate_semantic_similarity(summary, document)
        lexical_overlap_score = self.lexical_overlap(summary, document)
        rouge_scores=self.calculate_rouge_scores(summary,document)

        # Calculate the average of the three scores

        accuracy_score = (semantic_similarity_score  + lexical_overlap_score + rouge_scores) / 3


        return accuracy_score



    #######################

    ### Completeness ######

    #######################

    def extract_keywords(self,text: str) -> list:
        """Extracts keywords from a given text using part-of-speech tagging.

        Args:
            text (str): The text to extract keywords from.

        Returns:
            list: A list of keywords.
        """

        keywords = []
        for sentence in nltk.sent_tokenize(text):
            for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
                if pos == "NN":  # Check if the part of speech is a noun
                    keywords.append(self.porter_stemmer.stem(word))

        return keywords


    def extract_named_entities(self,text: str) -> list:
        """Extracts named entities from a given text using SpaCy.

        Args:
            text (str): The text to extract named entities from.

        Returns:
            list: A list of named entities.
        """
        nlp = spacy.load("en_core_web_sm")  # Load the SpaCy en_core_web_sm model
        doc = nlp(text)
        named_entities = [entity.text for entity in doc.ents]  # Extract named entities

        return named_entities


    def extract_key_sentences(self,text: str) -> list:
          """Extracts key sentences from a given text using Sentence2Vec.

          Args:
              text (str): The text to extract key sentences from.

          Returns:
              list: A list of key sentences.
          """

          vectorizer = Vectorizer()
          sentences = nltk.sent_tokenize(text.lower())
          vectorizer.run( sentences)
          np_vector=np.array(vectorizer.vectors)

          # Calculate sentence similarity matrix
          similarity_matrix = np.zeros((np_vector.shape[0], np_vector.shape[0]))
          for i, sentence_i in enumerate(sentences):
              for j, sentence_j in enumerate(sentences):
                  if i != j:
                      similarity_matrix[i, j] = cosine(np_vector[i], np_vector[j])

          # Calculate average similarities
          average_similarities = []
          for i in range(len(sentences)):
              current_similarity = np.mean([similarity_matrix[i, j] for j in range(len(sentences))])
              average_similarities.append(current_similarity)

          # Identify key sentences
          threshold = np.percentile(average_similarities, 70)  # Top 30% of similarity scores
          key_sentences = []
          for i, average_similarity in enumerate(average_similarities):
              if average_similarity >= threshold:
                  key_sentences.append(sentences[i])

          return key_sentences

    def evaluate_completeness( self, summary: str, document: str) -> float:
        """
        Evaluates the completeness of a summary based on keyword matching, named entity matching,
        and similarity with key sentences.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
            float: The completeness score.
        """
        keywords = self.extract_keywords(document)
        named_entities = self.extract_named_entities(document)
        key_sentences = self.extract_key_sentences(document)

        # Calculate keyword overlap score
        keyword_score = len(set(keywords) & set ([self.porter_stemmer.stem(w) for w in nltk.word_tokenize(summary)])) / len(set(keywords))

        # Calculate named entity overlap score
        named_entity_score = len(set(named_entities) & set ([w for w in nltk.word_tokenize(summary)])) / len(set(named_entities))


        # Calculate key sentence overlap score
        key_sentence_score = len(set(key_sentences) & set ([w for w in nltk.sent_tokenize(summary.lower())])) / len(set(key_sentences))


        # Calculate average overlap score
        completeness_score = (keyword_score + named_entity_score + key_sentence_score) / 3


        return completeness_score

    #######################

    ### Fluency ######

    #######################

    def check_grammatical_correctness(self,text: str) -> int:
        """Checks the grammatical correctness of a given text using NLTK's parse_sentence function.

        Args:
            text (str): The text to check for grammatical correctness.

        Returns:
            int: The number of grammatical errors found in the text.
        """
        errors = 0
        for sentence in nltk.sent_tokenize(text):
            try:
                nltk.parse.parse_sentence(sentence)
            except Exception:
                errors += 1

        return errors


    def calculate_readability(self,text: str) -> tuple:
        """Calculates the readability of a given text using Flesch-Kincaid and Gunning Fog indices.

        Args:
            text (str): The text to calculate the readability of.

        Returns:
            tuple: A tuple containing the Flesch-Kincaid grade level and Gunning Fog index.
        """
        readability_results = readability.getmeasures(text, lang='en')
        kincaid_score=readability_results['readability grades']['Kincaid']
        gunning_fog_index=readability_results['readability grades']['GunningFogIndex']


        return kincaid_score, gunning_fog_index


    def assess_naturalness(self,text: str) -> int:
        """Assesses the naturalness of a given text by counting idiomatic expressions and awkward phrasing.

        Args:
            text (str): The text to assess the naturalness of.

        Returns:
            int: A naturalness score, where higher scores indicate more natural language.
        """
        naturalness_score = 0

        # Check for idiomatic expressions
        idiomatic_expressions = ["on the other hand", "as a matter of fact", "in all likelihood"]
        for expression in idiomatic_expressions:
            if expression in text:
                naturalness_score += 1

        # Check for awkward or unnatural phrasing
        awkward_phrasing = ["I would like to...", "I hope that...", "It is important to..."]
        for phrase in awkward_phrasing:
            if phrase in text:
                naturalness_score -= 1

        return naturalness_score


    def evaluate_Fluency(self,summary: str, document: str) -> float:
        """
        Evaluates the fluency score of a summary based on grammatical correctness, readability, and naturalness.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
            float: The fluency score.
        """

        # Check grammatical correctness
        grammatical_errors = self.check_grammatical_correctness(summary)

        # Calculate readability
        readability_scores = self.calculate_readability(summary)
        flesch_kincaid_score = readability_scores[0]
        gunning_fog_index = readability_scores[1]

        # Assess naturalness
        naturalness_score = self.assess_naturalness(summary)

        # Combine readability scores
        readability_score = (flesch_kincaid_score + gunning_fog_index) / 2

        # Adjust readability score based on grammatical correctness
        if grammatical_errors == 0:
            readability_score *= 1.5

        # Calculate fluency score
        fluency_score = (readability_score + naturalness_score) / 2

        return fluency_score



    #######################

    ### Relevance ######

    #######################

    def calculate_word_overlap(self,summary: str, document: str) -> float:
        """
        Calculates the word overlap between a summary and a document.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
            float: The word overlap score.
        """
        words = set(nltk.word_tokenize(summary))
        document_words = set(nltk.word_tokenize(document))

        return len(words & document_words) / len(words)


    def calculate_tf_idf(self,text: str) -> np.array:
        """
        Calculates the TF-IDF scores for each word in a text.

        Args:
            text (str): The text to calculate TF-IDF scores for.

        Returns:
            np.array: The TF-IDF scores for each word.
        """
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text])

        return vectors.toarray()[0]




    def calculate_bleu_score(self, summary: str, document: str) -> float:

        """
        Calculates the bleu score for document and the generated summary.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
         float: bleu score.
        """

        # Tokenize the text and summary
        text_tokens = self.tokenizer(document, truncation=True, padding="max_length",return_tensors="pt").input_ids.to(self.device)
        summary_tokens = self.tokenizer(summary, truncation=True, padding="max_length",return_tensors="pt").input_ids.to(self.device)

        # Calculate the BLEU score
        with torch.no_grad():
            outputs = self.bert_model(text_tokens, attention_mask=torch.ones(1, len(text_tokens), dtype=torch.int64).to(self.device))
            pred = outputs.logits.argmax(-1)
            bleu = sentence_bleu([summary_tokens.cpu().numpy().tolist()[0]], pred.cpu().numpy().tolist()[0])

        return bleu




    def calculate_final_relevance_score(self,summary: str, document: str) -> float:
        """
        Calculates the final relevance score by combining word overlap, TF-IDF, and semantic similarity scores.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
            float: The final relevance score.
        """
        # Calculate the weighted average of the scores
        word_overlap_weight = 0.5
        tf_idf_weight = 0.5
       # bleu_weight=0.2 have to find workaround to check for tokens > 512



        # Calculate word overlap score
        word_overlap_score = self.calculate_word_overlap(summary, document)

        # Calculate TF-IDF score
        tf_idf_score = self.calculate_tf_idf(summary)

        # Calculate Bleu score
     #   bleu_scores=self.calculate_bleu_score(summary,document)


        final_relevance_score = (
            word_overlap_score * word_overlap_weight
            + tf_idf_score.mean() * tf_idf_weight
           # +  bleu_scores*bleu_weight

        )

        return final_relevance_score





    #######################

    ### Differentiability  ######

    #######################


    def calculate_jaccard_index_score(self, summary: str, document: str) -> float:

        """
        Calculates the jaccard_index score for document and the generated summary.

        Args:
            summary (str): The summary text.
            document (str): The original document text.

        Returns:
         float: jaccard_index score.
        """

        text_tokens = set(document.split())
        summary_tokens = set(summary.split())

        intersection = len(text_tokens & summary_tokens)
        union = len(text_tokens | summary_tokens)

        jaccard_index = intersection / union

        return jaccard_index






Summary_eval=Summarizer_evaluation()
"""

document = "I recently purchased this product and I am very happy with it. \
    It is easy to use and works great. I would definitely recommend this product to others. \
        It is a great value for the price and I am glad I bought it. \
            I have been using it for a few weeks now and I have no complaints. \
                It is exactly what I was looking for and I am very happy with my purchase.\
    I was disappointed with this product. It did not meet my expectations. \
    I would not recommend this product to others. It was difficult to use and did not work as expected. \
        I am not sure if I received a defective product or if this is just how the product is supposed to work. \
            I am not happy with my purchase and I would not recommend this product to anyone."



summary = "I recently purchased this product and I am very happy with it.\
    It is easy to use and works great. I would definitely recommend this product to others. \
        I was disappointed with this product. It did not meet my expectations.\
            I am not sure if I received a defective product or if this is just how the product is supposed to work."





Accuracy_score=Summary_eval.evaluate_accuracy(summary,document)
completeness_score=Summary_eval.evaluate_completeness(summary,document)
fluency_score=Summary_eval.evaluate_Fluency(summary,document)
Relevance_score=Summary_eval.calculate_final_relevance_score(summary,document)
Differentiability_score=Summary_eval.calculate_jaccard_index_score(summary,document)
"""
