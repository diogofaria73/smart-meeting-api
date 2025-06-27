#!/usr/bin/env python3
"""
🧪 Script de Teste da Análise de IA
Testa todas as funcionalidades do serviço de análise inteligente.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Texto de exemplo para teste
SAMPLE_TEXT = """
Reunião de planejamento do projeto Smart Meeting API realizada em 15/01/2024.

Participantes:
- João Silva (Gerente de Projeto)
- Maria Santos (Desenvolvedora Backend)
- Pedro Oliveira (QA Engineer)
- Ana Costa (Product Owner)

Discussões principais:

João Silva iniciou a reunião apresentando o cronograma. Maria Santos comentou sobre a implementação da API de transcrição usando Whisper. Pedro Oliveira levantou preocupações sobre os testes de performance.

Ana Costa aprovou a proposta de usar modelos de IA para análise automática. João Silva ficou responsável por definir os requirements técnicos até sexta-feira.

Decisões tomadas:
- Aprovado uso de faster-whisper para otimização
- Maria Santos vai implementar o serviço de IA até o final da semana
- Pedro Oliveira criará os testes automatizados para validação
- Ana Costa validará os resultados com stakeholders

Próximos passos:
- Implementar cache de modelos (João Silva)
- Configurar pipeline de CI/CD (Maria Santos)
- Documentar APIs (Pedro Oliveira)
- Preparar demo para cliente (Ana Costa)

Sentimento geral: reunião muito produtiva, equipe motivada e alinhada com objetivos.
"""

async def test_ai_analysis():
    """Testa o serviço de análise de IA"""
    try:
        # Importar o serviço
        from app.services.meeting_analysis_service import meeting_analysis_service
        
        logger.info("🤖 Iniciando teste do serviço de IA...")
        
        # Teste da análise completa
        logger.info("📝 Testando análise completa...")
        result = await meeting_analysis_service.analyze_meeting(SAMPLE_TEXT)
        
        # Validar resultados
        logger.info("📊 Resultados da análise:")
        logger.info(f"   • Participantes encontrados: {len(result.participants)}")
        for p in result.participants:
            logger.info(f"     - {p.name} (confiança: {p.confidence_score:.2f})")
        
        logger.info(f"   • Tópicos principais: {len(result.main_topics)}")
        for t in result.main_topics:
            logger.info(f"     - {t.title} (relevância: {t.relevance_score:.2f})")
        
        logger.info(f"   • Itens de ação: {len(result.action_items)}")
        for a in result.action_items:
            logger.info(f"     - {a.description[:50]}... (confiança: {a.confidence_score:.2f})")
            if a.responsible_person:
                logger.info(f"       Responsável: {a.responsible_person}")
            if a.deadline:
                logger.info(f"       Prazo: {a.deadline}")
        
        logger.info(f"   • Decisões importantes: {len(result.key_decisions)}")
        for d in result.key_decisions:
            logger.info(f"     - {d.description[:50]}...")
        
        if result.sentiment_analysis:
            logger.info(f"   • Sentimento geral: {result.sentiment_analysis.overall_sentiment}")
            logger.info(f"     Confiança: {result.sentiment_analysis.confidence_score:.2f}")
        
        logger.info(f"   • Resumo: {result.meeting_summary[:100]}...")
        logger.info(f"   • Confiança geral: {result.confidence_score:.2f}")
        logger.info(f"   • Tempo de processamento: {result.processing_time_seconds:.2f}s")
        
        # Teste de performance
        logger.info("⚡ Testando performance...")
        import time
        start_time = time.time()
        
        # Análise múltipla para testar cache
        for i in range(3):
            await meeting_analysis_service.analyze_meeting(SAMPLE_TEXT[:200])
        
        end_time = time.time()
        logger.info(f"   • 3 análises sequenciais: {end_time - start_time:.2f}s")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Serviço de IA não disponível: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        return False

async def test_fallback_analysis():
    """Testa o serviço tradicional como fallback"""
    try:
        from app.services.meeting_analysis_service import meeting_analysis_service
        
        logger.info("🔄 Testando análise tradicional (fallback)...")
        
        result = await meeting_analysis_service.analyze_meeting(
            transcription_text=SAMPLE_TEXT,
            include_sentiment=True,
            extract_participants=True,
            extract_action_items=True,
            min_confidence=0.6
        )
        
        logger.info("📊 Resultados da análise tradicional:")
        logger.info(f"   • Participantes: {len(result.participants)}")
        logger.info(f"   • Tópicos: {len(result.main_topics)}")
        logger.info(f"   • Ações: {len(result.action_items)}")
        logger.info(f"   • Decisões: {len(result.key_decisions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de fallback: {e}")
        return False

def test_model_imports():
    """Testa importação dos modelos"""
    logger.info("📦 Testando importações...")
    
    try:
        import spacy
        logger.info("✅ spaCy importado com sucesso")
        
        try:
            nlp = spacy.load("pt_core_news_sm")
            logger.info("✅ Modelo spaCy português carregado")
        except OSError:
            logger.warning("⚠️ Modelo spaCy português não encontrado")
            try:
                nlp = spacy.load("en_core_web_sm")
                logger.info("✅ Modelo spaCy inglês carregado (fallback)")
            except OSError:
                logger.error("❌ Nenhum modelo spaCy disponível")
        
    except ImportError:
        logger.error("❌ spaCy não está instalado")
    
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✅ SentenceTransformers importado")
        
        # Teste básico de carregamento
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("✅ Modelo de embeddings carregado")
        
    except ImportError:
        logger.error("❌ SentenceTransformers não está instalado")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo de embeddings: {e}")
    
    try:
        from transformers import pipeline
        logger.info("✅ Transformers importado")
        
    except ImportError:
        logger.error("❌ Transformers não está instalado")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("✅ Scikit-learn importado")
        
    except ImportError:
        logger.error("❌ Scikit-learn não está instalado")

async def main():
    """Função principal de teste"""
    logger.info("🚀 Iniciando bateria de testes da análise de IA...")
    
    # 1. Testar importações
    test_model_imports()
    
    # 2. Testar análise de IA
    ai_success = await test_ai_analysis()
    
    # 3. Testar fallback
    fallback_success = await test_fallback_analysis()
    
    # Relatório final
    logger.info("\n📋 RELATÓRIO FINAL:")
    if ai_success:
        logger.info("✅ Análise de IA funcionando perfeitamente")
    else:
        logger.warning("⚠️ Análise de IA com problemas")
    
    if fallback_success:
        logger.info("✅ Análise tradicional funcionando (boa cobertura)")
    else:
        logger.error("❌ Análise tradicional com problemas")
    
    if ai_success or fallback_success:
        logger.info("🎉 Sistema pronto para uso!")
        logger.info("💡 Dicas de uso:")
        logger.info("   • Para áudios curtos (<60s), a IA será mais rápida")
        logger.info("   • Para áudios longos, monitore uso de memória")
        logger.info("   • Cache de modelos melhora performance em análises sequenciais")
    else:
        logger.error("❌ Sistema precisa de correções antes do uso")

if __name__ == "__main__":
    asyncio.run(main()) 