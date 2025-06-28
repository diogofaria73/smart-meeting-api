#!/usr/bin/env python3
"""
🎙️ SCRIPT DE INSTALAÇÃO - SPEAKER DIARIZATION
Instala e configura as dependências necessárias para identificação de speakers
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """Executa um comando e retorna se foi bem-sucedido"""
    try:
        logger.info(f"🔄 {description}")
        logger.info(f"   Comando: {command}")
        
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ {description} - Concluído")
        if result.stdout:
            logger.debug(f"   Saída: {result.stdout[:200]}...")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - Falhou")
        logger.error(f"   Código de erro: {e.returncode}")
        logger.error(f"   Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ {description} - Erro inesperado: {e}")
        return False


def check_python_version():
    """Verifica se a versão do Python é compatível"""
    logger.info("🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error(f"❌ Python {version.major}.{version.minor} não é suportado")
        logger.error("   Requisito: Python 3.10 ou superior")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} é compatível")
    return True


def check_poetry():
    """Verifica se Poetry está instalado"""
    logger.info("📦 Verificando Poetry...")
    
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Poetry disponível: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Poetry não encontrado")
            return False
    except FileNotFoundError:
        logger.error("❌ Poetry não está instalado")
        logger.info("   Instale com: curl -sSL https://install.python-poetry.org | python3 -")
        return False


def install_dependencies():
    """Instala as dependências via Poetry"""
    logger.info("📦 Instalando dependências do projeto...")
    
    # Instala dependências principais
    if not run_command("poetry install", "Instalando dependências principais"):
        return False
    
    # Verifica se pyannote.audio foi instalado corretamente
    try:
        logger.info("🔍 Verificando instalação do pyannote.audio...")
        result = subprocess.run(
            ["poetry", "run", "python", "-c", "import pyannote.audio; print('✅ pyannote.audio importado com sucesso')"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(result.stdout.strip())
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ Erro ao verificar pyannote.audio")
        logger.error(f"   Stderr: {e.stderr}")
        return False


def create_env_template():
    """Cria template de variáveis de ambiente"""
    logger.info("📝 Criando template de configuração...")
    
    env_template = """
# 🎙️ CONFIGURAÇÕES DE SPEAKER DIARIZATION

# Token do HuggingFace Hub (opcional, mas recomendado)
# Obtenha em: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=your_token_here

# Configurações de diarização
ENABLE_SPEAKER_DIARIZATION=true
MIN_SPEAKERS=1
MAX_SPEAKERS=10
MIN_SEGMENT_DURATION=1.0

# Configurações de hardware
# Deixe vazio para detecção automática
FORCE_DEVICE=  # cuda, mps, ou cpu
FORCE_COMPUTE_TYPE=  # float16, float32, ou int8
"""
    
    env_file = Path(".env.diarization")
    
    try:
        with open(env_file, "w") as f:
            f.write(env_template.strip())
        
        logger.info(f"✅ Template criado: {env_file}")
        logger.info("   Configure suas variáveis de ambiente neste arquivo")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar template: {e}")
        return False


def test_diarization():
    """Testa se a diarização está funcionando"""
    logger.info("🧪 Testando funcionalidade de diarização...")
    
    test_script = """
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    from app.services.speaker_diarization_service import speaker_diarization_service
    info = speaker_diarization_service.get_performance_info()
    
    print('✅ Serviço de diarização carregado com sucesso')
    print(f'   - pyannote disponível: {info["pyannote_available"]}')
    print(f'   - Device: {info["device"]}')
    print(f'   - Pipeline carregado: {info["pipeline_loaded"]}')
    
    if info["pyannote_available"]:
        print('🎉 Tudo pronto para identificação de speakers!')
    else:
        print('⚠️ pyannote.audio não disponível')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Erro no teste: {e}')
    sys.exit(1)
"""
    
    try:
        result = subprocess.run(
            ["poetry", "run", "python", "-c", test_script],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            logger.info("✅ Teste de diarização passou")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"   {line}")
            return True
        else:
            logger.error("❌ Teste de diarização falhou")
            logger.error(f"   Stdout: {result.stdout}")
            logger.error(f"   Stderr: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao executar teste: {e}")
        return False


def print_next_steps():
    """Imprime próximos passos"""
    logger.info("\n" + "="*60)
    logger.info("🎉 INSTALAÇÃO DA DIARIZAÇÃO CONCLUÍDA!")
    logger.info("="*60)
    
    logger.info("\n📋 PRÓXIMOS PASSOS:")
    logger.info("   1. Configure seu token do HuggingFace (opcional):")
    logger.info("      - Acesse: https://huggingface.co/settings/tokens")
    logger.info("      - Aceite os termos: https://huggingface.co/pyannote/speaker-diarization-3.1")
    logger.info("      - Adicione o token no arquivo .env.diarization")
    
    logger.info("\n   2. Inicie a API:")
    logger.info("      poetry run python run.py")
    
    logger.info("\n   3. Teste a diarização:")
    logger.info("      - Endpoint: POST /api/transcriptions/test-diarization")
    logger.info("      - Envie um arquivo de áudio com múltiplos speakers")
    logger.info("      - Verifique os speakers identificados na resposta")
    
    logger.info("\n🎙️ FUNCIONALIDADES DISPONÍVEIS:")
    logger.info("   ✅ Identificação automática de speakers")
    logger.info("   ✅ Segmentação temporal por speaker")
    logger.info("   ✅ Estatísticas de participação")
    logger.info("   ✅ Integração com transcrição Whisper")
    logger.info("   ✅ Fallback para transcrição simples")
    
    logger.info("\n💡 DICAS:")
    logger.info("   - Áudios com >30s funcionam melhor para diarização")
    logger.info("   - GPU com 6GB+ recomendada para performance")
    logger.info("   - Suporte a formatos: MP3, WAV, M4A, FLAC, OGG")


def main():
    """Função principal de instalação"""
    logger.info("🎙️ INSTALADOR DE SPEAKER DIARIZATION")
    logger.info("="*60)
    
    # Verificações básicas
    if not check_python_version():
        sys.exit(1)
    
    if not check_poetry():
        sys.exit(1)
    
    # Instalação
    if not install_dependencies():
        logger.error("❌ Falha na instalação das dependências")
        sys.exit(1)
    
    # Configuração
    create_env_template()
    
    # Teste
    if not test_diarization():
        logger.warning("⚠️ Teste inicial falhou, mas instalação pode estar OK")
        logger.warning("   Verifique se há problemas de configuração")
    
    # Conclusão
    print_next_steps()


if __name__ == "__main__":
    main() 