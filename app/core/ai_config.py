# Configuração dos Modelos de IA
# Este arquivo é gerado automaticamente

# Modelos principais
SPACY_MODEL = "pt_core_news_sm"
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
NER_MODEL = "neuralmind/bert-base-portuguese-cased"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Configurações de performance
USE_GPU = True
CACHE_DIR = "./models_cache"
MAX_SEQUENCE_LENGTH = 512

# Thresholds
SIMILARITY_THRESHOLD = 0.80
CONFIDENCE_THRESHOLD = 0.75
