.PHONY: install run test format lint docker-up docker-down docker-build docker-run docker-all setup env-setup db-setup prisma-init prisma-generate prisma-migrate prisma-studio

# Instalação de dependências
install:
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
	docker-compose up -d db

docker-down:
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
	poetry run prisma db push

prisma-generate:
	poetry run prisma generate

prisma-migrate:
	poetry run prisma migrate dev

prisma-studio:
	poetry run prisma studio

# Setup inicial
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
	@echo "🤖 Configurando modelos de IA..."
	poetry run python scripts/setup_ai_models.py

test-ai:
	@echo "🧪 Testando análise de IA..."
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

# Setup completo com IA
setup-full: setup setup-ai
	@echo "🎉 Setup completo com IA finalizado!" 