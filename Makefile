.PHONY: install run test format lint docker-up docker-down docker-build docker-run docker-all setup env-setup db-setup prisma-init prisma-generate prisma-migrate prisma-studio setup-first-time setup-env-files check-requirements

# Cores para output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
BOLD := \033[1m
NC := \033[0m # No Color

# ==============================================
# COMANDOS DE SETUP INICIAL
# ==============================================

# Setup completo para primeira execução
setup-first-time: check-requirements setup-env-files docker-up wait-db install prisma-generate prisma-init setup-ai install-diarization-deps
	@echo ""
	@echo "$(GREEN)$(BOLD)🎉 SETUP INICIAL CONCLUÍDO COM SUCESSO! 🎉$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(GREEN)✅ Infraestrutura PostgreSQL configurada$(NC)"
	@echo "$(GREEN)✅ Arquivos de configuração (.env) criados$(NC)"
	@echo "$(GREEN)✅ Dependências Python instaladas$(NC)"
	@echo "$(GREEN)✅ Banco de dados inicializado$(NC)"
	@echo "$(GREEN)✅ Modelos de IA configurados$(NC)"
	@echo "$(GREEN)✅ Dependências de diarização instaladas$(NC)"
	@echo ""
	@echo "$(YELLOW)🚀 Para iniciar a aplicação:$(NC)"
	@echo "   make run"
	@echo ""
	@echo "$(YELLOW)📊 Para acessar o Prisma Studio:$(NC)"
	@echo "   make prisma-studio"
	@echo ""
	@echo "$(YELLOW)🧪 Para testar a IA:$(NC)"
	@echo "   make test-ai"

# Verifica se os requisitos estão instalados
check-requirements:
	@echo "$(BLUE)🔍 Verificando requisitos do sistema...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)❌ Docker não encontrado. Instale o Docker primeiro.$(NC)"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)❌ Docker Compose não encontrado.$(NC)"; exit 1; }
	@command -v poetry >/dev/null 2>&1 || { echo "$(RED)❌ Poetry não encontrado. Instale com: curl -sSL https://install.python-poetry.org | python3 -$(NC)"; exit 1; }
	@python3 --version >/dev/null 2>&1 || { echo "$(RED)❌ Python 3 não encontrado.$(NC)"; exit 1; }
	@echo "$(GREEN)✅ Todos os requisitos estão instalados$(NC)"

# Cria todos os arquivos de ambiente necessários
setup-env-files:
	@echo "$(BLUE)📝 Criando arquivos de configuração...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Criando arquivo .env principal...$(NC)"; \
		echo '# ===========================================' > .env; \
		echo '# CONFIGURAÇÕES SMART MEETING API' >> .env; \
		echo '# ===========================================' >> .env; \
		echo '' >> .env; \
		echo '# Configurações gerais' >> .env; \
		echo 'PROJECT_NAME="Smart Meeting API"' >> .env; \
		echo 'API_PREFIX="/api"' >> .env; \
		echo 'SECRET_KEY="seu_secret_key_super_seguro_aqui_$(shell date +%s)"' >> .env; \
		echo 'DEBUG=true' >> .env; \
		echo '' >> .env; \
		echo '# Configurações de CORS' >> .env; \
		echo 'ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"' >> .env; \
		echo '' >> .env; \
		echo '# Configurações do banco de dados' >> .env; \
		echo 'DATABASE_URL="postgresql://postgres:postgres@localhost:5433/smart_meeting"' >> .env; \
		echo '' >> .env; \
		echo '# Configurações para modelos de IA' >> .env; \
		echo 'MODEL_PATH="facebook/wav2vec2-large-960h"' >> .env; \
		echo 'SUMMARIZATION_MODEL="facebook/bart-large-cnn"' >> .env; \
		echo 'SPACY_MODEL="pt_core_news_sm"' >> .env; \
		echo '' >> .env; \
		echo '# Configurações de upload' >> .env; \
		echo 'MAX_FILE_SIZE=100' >> .env; \
		echo 'UPLOAD_DIR="temp_audio"' >> .env; \
		echo '' >> .env; \
		echo '# Configurações de WebSocket' >> .env; \
		echo 'WEBSOCKET_PING_INTERVAL=30' >> .env; \
		echo 'WEBSOCKET_PING_TIMEOUT=10' >> .env; \
		echo "$(GREEN)✅ .env criado com sucesso!$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ .env já existe. Mantendo configurações atuais.$(NC)"; \
	fi
	@if [ ! -f .env.diarization ]; then \
		echo "$(YELLOW)Criando arquivo .env.diarization...$(NC)"; \
		echo '# ===========================================' > .env.diarization; \
		echo '# CONFIGURAÇÕES DE SPEAKER DIARIZATION' >> .env.diarization; \
		echo '# ===========================================' >> .env.diarization; \
		echo '' >> .env.diarization; \
		echo '# Token do HuggingFace Hub (opcional, mas recomendado)' >> .env.diarization; \
		echo '# Obtenha em: https://huggingface.co/settings/tokens' >> .env.diarization; \
		echo 'HUGGINGFACE_TOKEN=your_token_here' >> .env.diarization; \
		echo '' >> .env.diarization; \
		echo '# Configurações de diarização' >> .env.diarization; \
		echo 'ENABLE_SPEAKER_DIARIZATION=true' >> .env.diarization; \
		echo 'MIN_SPEAKERS=1' >> .env.diarization; \
		echo 'MAX_SPEAKERS=10' >> .env.diarization; \
		echo 'MIN_SEGMENT_DURATION=1.0' >> .env.diarization; \
		echo '' >> .env.diarization; \
		echo '# Configurações de hardware' >> .env.diarization; \
		echo 'FORCE_DEVICE=  # cuda, mps, ou cpu' >> .env.diarization; \
		echo 'FORCE_COMPUTE_TYPE=  # float16, float32, ou int8' >> .env.diarization; \
		echo "$(GREEN)✅ .env.diarization criado com sucesso!$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ .env.diarization já existe. Mantendo configurações atuais.$(NC)"; \
	fi

# Aguarda o banco de dados ficar pronto
wait-db:
	@echo "$(BLUE)⏳ Aguardando banco de dados PostgreSQL ficar pronto...$(NC)"
	@timeout=30; \
	while [ $$timeout -gt 0 ]; do \
		if docker exec smart-meeting-db pg_isready -U postgres >/dev/null 2>&1; then \
			echo "$(GREEN)✅ Banco de dados PostgreSQL está pronto!$(NC)"; \
			break; \
		fi; \
		echo "$(YELLOW)⏳ Aguardando... ($$timeout segundos restantes)$(NC)"; \
		sleep 2; \
		timeout=$$((timeout - 2)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "$(RED)❌ Timeout: Banco de dados não ficou pronto em 30 segundos$(NC)"; \
		exit 1; \
	fi

# Instala dependências de diarização
install-diarization-deps:
	@echo "$(BLUE)🎙️ Instalando dependências de diarização...$(NC)"
	@poetry run python scripts/install_diarization.py

# ==============================================
# COMANDOS EXISTENTES (mantidos)
# ==============================================

# Instalação de dependências
install:
	@echo "$(BLUE)📦 Instalando dependências Python...$(NC)"
	poetry install

# Execução da aplicação
run:
	poetry run uvicorn app.main:app --reload

# Execução de testes
test:
	poetry run pytest

# Formatação de código
format:
	poetry run black .
	poetry run isort .

# Verificação de tipos
lint:
	poetry run mypy .

# Docker - Banco de dados
docker-up:
	@echo "$(BLUE)🐳 Iniciando infraestrutura PostgreSQL...$(NC)"
	docker-compose up -d db
	@echo "$(GREEN)✅ PostgreSQL iniciado em background$(NC)"

docker-down:
	@echo "$(BLUE)🛑 Parando infraestrutura Docker...$(NC)"
	docker-compose down

# Docker - Aplicação
docker-build:
	docker-compose build api

docker-run: docker-up
	docker-compose up api

docker-all: docker-up
	docker-compose up -d

# Prisma
prisma-init:
	@echo "$(BLUE)🗄️ Inicializando esquema do banco de dados...$(NC)"
	poetry run prisma db push

prisma-generate:
	@echo "$(BLUE)⚙️ Gerando cliente Prisma...$(NC)"
	poetry run prisma generate

prisma-migrate:
	poetry run prisma migrate dev

prisma-studio:
	poetry run prisma studio

# Setup inicial (versão básica - mantida para compatibilidade)
env-setup:
	@echo "Criando arquivo .env..."
	@if [ ! -f .env ]; then \
		echo '# Configurações gerais\nPROJECT_NAME="Smart Meeting API"\nAPI_PREFIX="/api"\nSECRET_KEY="seu_secret_key_seguro_aqui"\n\n# Configurações de CORS\nALLOWED_ORIGINS="http://localhost:3000,http://localhost:8000"\n\n# Configurações do banco de dados\nDATABASE_URL="postgresql://postgres:postgres@localhost:5433/smart_meeting"\n\n# Configurações para modelos de IA\nMODEL_PATH="facebook/wav2vec2-large-960h"\nSUMMARIZATION_MODEL="facebook/bart-large-cnn"' > .env; \
		echo ".env criado com sucesso!"; \
	else \
		echo ".env já existe. Pulando..."; \
	fi

db-setup: docker-up
	@echo "Aguardando o banco de dados iniciar..."
	@sleep 5
	@echo "Banco de dados pronto!"

# Configuração de IA
setup-ai:
	@echo "$(BLUE)🤖 Configurando modelos de IA...$(NC)"
	poetry run python scripts/setup_ai_models.py

test-ai:
	@echo "$(BLUE)🧪 Testando análise de IA...$(NC)"
	poetry run python scripts/test_ai_analysis.py

# Performance check
check-performance:
	@echo "⚡ Verificando performance do sistema..."
	@echo "GPU disponível:" && python -c "import torch; print('✅ Sim' if torch.cuda.is_available() else '❌ Não')"
	@echo "Memória RAM:" && python -c "import psutil; print(f'{psutil.virtual_memory().total / 1024**3:.1f}GB')"
	@echo "CPU cores:" && python -c "import os; print(os.cpu_count())"

setup: install env-setup db-setup prisma-generate prisma-init
	@echo "Setup inicial concluído!"
	@echo "🚀 Para configurar a IA, execute: make setup-ai"

# Setup completo com IA (versão básica - mantida para compatibilidade)
setup-full: setup setup-ai
	@echo "🎉 Setup completo com IA finalizado!"

# ==============================================
# COMANDOS DE AJUDA
# ==============================================

help:
	@echo "$(BOLD)Smart Meeting API - Comandos Make$(NC)"
	@echo ""
	@echo "$(YELLOW)SETUP INICIAL (primeira vez):$(NC)"
	@echo "  setup-first-time     - Setup completo para primeira execução"
	@echo ""
	@echo "$(YELLOW)DESENVOLVIMENTO:$(NC)"
	@echo "  run                  - Executa a aplicação"
	@echo "  test                 - Executa testes"
	@echo "  format               - Formata código"
	@echo "  lint                 - Verifica tipos"
	@echo ""
	@echo "$(YELLOW)DOCKER:$(NC)"
	@echo "  docker-up            - Inicia PostgreSQL"
	@echo "  docker-down          - Para Docker"
	@echo ""
	@echo "$(YELLOW)BANCO DE DADOS:$(NC)"
	@echo "  prisma-studio        - Interface gráfica do banco"
	@echo "  prisma-migrate       - Cria nova migration"
	@echo ""
	@echo "$(YELLOW)IA:$(NC)"
	@echo "  setup-ai             - Configura modelos de IA"
	@echo "  test-ai              - Testa análise de IA" 