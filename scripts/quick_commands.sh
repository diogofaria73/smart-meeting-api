#!/bin/bash

# 🎙️ COMANDOS RÁPIDOS - SPEAKER DIARIZATION
# Comandos úteis para instalar, configurar e testar a funcionalidade

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Funções auxiliares
print_header() {
    echo -e "${CYAN}🎙️ $1${NC}"
    echo "========================================"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Função para verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. INSTALAÇÃO COMPLETA
install() {
    print_header "INSTALAÇÃO COMPLETA DA DIARIZAÇÃO"
    
    # Verificar Poetry
    if ! command_exists poetry; then
        print_error "Poetry não encontrado!"
        print_info "Instale com: curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi
    
    # Instalar dependências
    print_info "Instalando dependências..."
    poetry install
    
    # Executar script de instalação
    if [ -f "scripts/install_diarization.py" ]; then
        print_info "Executando script de configuração..."
        python scripts/install_diarization.py
    else
        print_warning "Script de instalação não encontrado"
    fi
    
    print_success "Instalação concluída!"
}

# 2. TESTE RÁPIDO
test() {
    print_header "TESTE RÁPIDO DA DIARIZAÇÃO"
    
    # Verificar se a API está rodando
    if ! curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        print_error "API não está rodando!"
        print_info "Inicie com: poetry run python run.py"
        exit 1
    fi
    
    # Verificar se há arquivo de teste
    if [ ! -f "test_audio.wav" ] && [ ! -f "test_audio.mp3" ]; then
        print_warning "Nenhum arquivo de teste encontrado (test_audio.wav/mp3)"
        print_info "Adicione um arquivo de teste ou use: test_file <caminho_do_arquivo>"
        return 1
    fi
    
    # Usar arquivo disponível
    TEST_FILE=""
    if [ -f "test_audio.wav" ]; then
        TEST_FILE="test_audio.wav"
    elif [ -f "test_audio.mp3" ]; then
        TEST_FILE="test_audio.mp3"
    fi
    
    print_info "Testando com arquivo: $TEST_FILE"
    
    # Fazer requisição de teste
    curl -X POST "http://localhost:8000/api/transcriptions/test-diarization" \
        -F "file=@$TEST_FILE" \
        -F "enable_diarization=true" \
        -H "accept: application/json" | jq '.'
    
    print_success "Teste concluído!"
}

# 3. TESTE COM ARQUIVO ESPECÍFICO
test_file() {
    if [ -z "$1" ]; then
        print_error "Uso: $0 test_file <caminho_do_arquivo>"
        exit 1
    fi
    
    local FILE_PATH="$1"
    
    if [ ! -f "$FILE_PATH" ]; then
        print_error "Arquivo não encontrado: $FILE_PATH"
        exit 1
    fi
    
    print_header "TESTE COM ARQUIVO ESPECÍFICO"
    print_info "Arquivo: $FILE_PATH"
    
    # Verificar se a API está rodando
    if ! curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        print_error "API não está rodando!"
        print_info "Inicie com: poetry run python run.py"
        exit 1
    fi
    
    # Fazer requisição
    curl -X POST "http://localhost:8000/api/transcriptions/test-diarization" \
        -F "file=@$FILE_PATH" \
        -F "enable_diarization=true" \
        -H "accept: application/json" | jq '.'
    
    print_success "Teste concluído!"
}

# 4. VERIFICAR STATUS
status() {
    print_header "STATUS DO SISTEMA"
    
    # Verificar Python
    echo -n "🐍 Python: "
    if command_exists python3; then
        python3 --version
    else
        print_error "Python3 não encontrado"
    fi
    
    # Verificar Poetry
    echo -n "📦 Poetry: "
    if command_exists poetry; then
        poetry --version
    else
        print_error "Poetry não encontrado"
    fi
    
    # Verificar dependências
    echo "🔍 Verificando dependências críticas..."
    poetry run python -c "
import sys
dependencies = ['torch', 'torchaudio', 'pyannote.audio', 'faster_whisper']
for dep in dependencies:
    try:
        __import__(dep.replace('-', '_'))
        print(f'  ✅ {dep}')
    except ImportError:
        print(f'  ❌ {dep}')
"
    
    # Verificar API
    echo -n "🌐 API Status: "
    if curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        print_success "Rodando (http://localhost:8000)"
    else
        print_warning "Não está rodando"
    fi
    
    # Verificar GPU
    echo "🖥️ Hardware:"
    poetry run python -c "
import torch
print(f'  - CUDA disponível: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  - GPU: {torch.cuda.get_device_name()}')
    print(f'  - VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
else:
    print('  - Usando CPU')
"
}

# 5. INICIAR API
start() {
    print_header "INICIANDO API"
    
    # Verificar se já está rodando
    if curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        print_warning "API já está rodando em http://localhost:8000"
        return 0
    fi
    
    print_info "Iniciando servidor..."
    poetry run python run.py
}

# 6. LOGS EM TEMPO REAL
logs() {
    print_header "MONITORAMENTO DE LOGS"
    
    if [ -f "logs/app.log" ]; then
        print_info "Monitorando logs de diarização..."
        tail -f logs/app.log | grep -E "(🎙️|speaker|diarization|SPEAKER_|pyannote)" --color=always
    else
        print_warning "Arquivo de log não encontrado: logs/app.log"
        print_info "Inicie a API primeiro com: $0 start"
    fi
}

# 7. LIMPAR CACHE
clean() {
    print_header "LIMPEZA DE CACHE"
    
    # Limpar cache Python
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Limpar cache de modelos (se existir)
    if [ -d "~/.cache/huggingface" ]; then
        print_info "Limpando cache HuggingFace..."
        du -sh ~/.cache/huggingface 2>/dev/null || true
    fi
    
    # Limpar arquivos temporários
    find . -name "*.tmp" -delete 2>/dev/null || true
    find temp_audio/ -name "*" -delete 2>/dev/null || true
    
    print_success "Limpeza concluída!"
}

# 8. BENCHMARK RÁPIDO
benchmark() {
    print_header "BENCHMARK DE PERFORMANCE"
    
    # Verificar se há arquivo de teste
    if [ ! -f "test_audio.wav" ] && [ ! -f "test_audio.mp3" ]; then
        print_warning "Nenhum arquivo de teste encontrado"
        print_info "Adicione test_audio.wav ou test_audio.mp3 para benchmark"
        return 1
    fi
    
    # Usar arquivo disponível
    TEST_FILE=""
    if [ -f "test_audio.wav" ]; then
        TEST_FILE="test_audio.wav"
    elif [ -f "test_audio.mp3" ]; then
        TEST_FILE="test_audio.mp3"
    fi
    
    print_info "Executando benchmark com: $TEST_FILE"
    
    # Teste sem diarização
    echo "📊 Teste 1: Transcrição simples"
    time curl -s -X POST "http://localhost:8000/api/transcriptions/test-diarization" \
        -F "file=@$TEST_FILE" \
        -F "enable_diarization=false" | jq -r '.processing.total_time // "N/A"'
    
    echo "📊 Teste 2: Com diarização"
    time curl -s -X POST "http://localhost:8000/api/transcriptions/test-diarization" \
        -F "file=@$TEST_FILE" \
        -F "enable_diarization=true" | jq -r '.processing.total_time // "N/A"'
    
    print_success "Benchmark concluído!"
}

# 9. AJUDA
help() {
    print_header "COMANDOS DISPONÍVEIS"
    
    echo "📋 Uso: $0 <comando>"
    echo ""
    echo "Comandos principais:"
    echo "  install        - Instala todas as dependências"
    echo "  test          - Teste rápido com arquivo padrão"
    echo "  test_file <file> - Teste com arquivo específico"
    echo "  status        - Verifica status do sistema"
    echo "  start         - Inicia a API"
    echo ""
    echo "Comandos de desenvolvimento:"
    echo "  logs          - Monitora logs em tempo real"
    echo "  clean         - Limpa cache e arquivos temporários"
    echo "  benchmark     - Executa benchmark de performance"
    echo "  help          - Mostra esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 install"
    echo "  $0 test_file reuniao.mp3"
    echo "  $0 status"
    echo ""
    print_info "Para mais detalhes, consulte: README_DIARIZATION.md"
}

# MAIN - Processar argumentos
case "$1" in
    "install")
        install
        ;;
    "test")
        test
        ;;
    "test_file")
        test_file "$2"
        ;;
    "status")
        status
        ;;
    "start")
        start
        ;;
    "logs")
        logs
        ;;
    "clean")
        clean
        ;;
    "benchmark")
        benchmark
        ;;
    "help"|"")
        help
        ;;
    *)
        print_error "Comando desconhecido: $1"
        echo ""
        help
        exit 1
        ;;
esac 