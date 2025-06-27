#!/usr/bin/env python3
"""
🤖 Script de Configuração dos Modelos de IA
Baixa e configura todos os modelos necessários para análise inteligente.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> bool:
    """Executa comando e retorna True se bem-sucedido"""
    try:
        logger.info(f"🔄 {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} - Concluído")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - Falhou: {e.stderr}")
        return False

def install_spacy_model():
    """Instala modelo spaCy para português"""
    logger.info("📦 Instalando modelo spaCy português...")
    
    # Primeiro, instalar o modelo português pequeno
    commands = [
        ("python -m spacy download pt_core_news_sm", "Download modelo português pequeno"),
        ("python -m spacy download pt_core_news_md", "Download modelo português médio (opcional)"),
    ]
    
    success = False
    for command, desc in commands:
        if run_command(command, desc):
            success = True
            break
    
    if not success:
        logger.warning("⚠️ Não foi possível instalar modelos spaCy em português")
        logger.info("🔄 Tentando instalar modelo inglês como fallback...")
        run_command("python -m spacy download en_core_web_sm", "Download modelo inglês (fallback)")

def verify_transformers_models():
    """Verifica e pré-carrega modelos Transformers"""
    logger.info("🔍 Verificando modelos Transformers...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
        from sentence_transformers import SentenceTransformer
        
        # Modelos para teste
        models_to_test = [
            ("neuralmind/bert-base-portuguese-cased", "BERT português"),
            ("paraphrase-multilingual-MiniLM-L12-v2", "SentenceTransformer multilíngue"),
            ("cardiffnlp/twitter-roberta-base-sentiment-latest", "Análise de sentimento")
        ]
        
        for model_name, description in models_to_test:
            try:
                logger.info(f"📥 Baixando {description}...")
                
                if "sentence" in model_name.lower() or "paraphrase" in model_name.lower():
                    # SentenceTransformer
                    model = SentenceTransformer(model_name)
                    logger.info(f"✅ {description} carregado com sucesso")
                else:
                    # Transformers regular
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    logger.info(f"✅ {description} tokenizer carregado")
                    
                    # Apenas testa o carregamento, não mantém em memória
                    try:
                        model = AutoModelForTokenClassification.from_pretrained(model_name)
                        logger.info(f"✅ {description} modelo carregado")
                        del model  # Libera memória
                    except:
                        model = AutoModel.from_pretrained(model_name)
                        logger.info(f"✅ {description} modelo base carregado")
                        del model  # Libera memória
                        
                del tokenizer
                
            except Exception as e:
                logger.warning(f"⚠️ Não foi possível carregar {description}: {e}")
                
    except ImportError as e:
        logger.error(f"❌ Erro de importação: {e}")
        return False
    
    return True

def install_additional_dependencies():
    """Instala dependências adicionais se necessário"""
    logger.info("📦 Verificando dependências adicionais...")
    
    optional_packages = [
        ("pip install --upgrade scikit-learn", "Atualizar scikit-learn"),
        ("pip install --upgrade numpy", "Atualizar numpy"),
    ]
    
    for command, desc in optional_packages:
        run_command(command, desc)

def create_model_config():
    """Cria arquivo de configuração dos modelos"""
    config_content = '''# Configuração dos Modelos de IA
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
'''
    
    config_path = Path("app/core/ai_config.py")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info(f"📄 Configuração salva em {config_path}")

def main():
    """Função principal"""
    logger.info("🚀 Iniciando configuração dos modelos de IA...")
    
    try:
        # 1. Instalar modelo spaCy
        install_spacy_model()
        
        # 2. Verificar modelos Transformers
        verify_transformers_models()
        
        # 3. Instalar dependências adicionais
        install_additional_dependencies()
        
        # 4. Criar configuração
        create_model_config()
        
        logger.info("🎉 Configuração dos modelos de IA concluída com sucesso!")
        logger.info("📝 Próximos passos:")
        logger.info("   1. Reinicie a aplicação")
        logger.info("   2. Teste um áudio pequeno primeiro")
        logger.info("   3. Monitore logs para verificar se os modelos carregaram")
        
    except Exception as e:
        logger.error(f"❌ Erro durante configuração: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 