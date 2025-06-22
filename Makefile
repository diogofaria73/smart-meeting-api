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

setup: install env-setup db-setup prisma-generate prisma-init
	@echo "Setup inicial concluído!" 