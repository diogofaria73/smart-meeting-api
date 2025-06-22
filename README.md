# Smart Meeting API

API para transcrição e resumo automático de reuniões utilizando transformers e redes neurais recorrentes.

## Tecnologias

- FastAPI
- Poetry
- Prisma
- PostgreSQL (Docker)
- Transformers (Hugging Face)
- PyTorch

## Requisitos

- Python 3.10+
- Poetry
- Docker e Docker Compose
- Make (opcional, para facilitar comandos)

## Instalação

### Setup Automático

O projeto inclui um comando para configuração automática:

```bash
make setup
```

Este comando irá:
1. Instalar as dependências com Poetry
2. Criar o arquivo `.env` (se não existir)
3. Iniciar o banco de dados PostgreSQL com Docker
4. Gerar o cliente Prisma
5. Inicializar o banco de dados

### Setup Manual

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/smart-meeting-api.git
cd smart-meeting-api
```

2. Instale as dependências:

```bash
make install
# ou
poetry install
```

3. Configure as variáveis de ambiente:

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```
# Configurações gerais
PROJECT_NAME="Smart Meeting API"
API_PREFIX="/api"
SECRET_KEY="seu_secret_key_seguro_aqui"

# Configurações de CORS
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8000"

# Configurações do banco de dados
DATABASE_URL="postgresql://postgres:postgres@localhost:5433/smart_meeting"

# Configurações para modelos de IA
MODEL_PATH="facebook/wav2vec2-large-960h"
SUMMARIZATION_MODEL="facebook/bart-large-cnn"
```

4. Inicie o banco de dados:

```bash
make docker-up
# ou
docker-compose up -d db
```

5. Inicialize o banco de dados:

```bash
make prisma-generate
make prisma-init
# ou
poetry run prisma generate
poetry run prisma db push
```

## Executando a aplicação

### Execução Local

```bash
make run
# ou
poetry run uvicorn app.main:app --reload
```

### Execução com Docker

O projeto também pode ser executado completamente em Docker:

```bash
# Construir a imagem da aplicação
make docker-build

# Executar apenas o banco de dados
make docker-up

# Executar a aplicação (e o banco de dados, se não estiver rodando)
make docker-run

# Executar todos os serviços em modo detached
make docker-all

# Parar todos os serviços
make docker-down
```

### Prisma Studio

Para visualizar e editar os dados do banco de dados através de uma interface gráfica, você pode usar o Prisma Studio:

```bash
make prisma-studio
# ou
poetry run prisma studio
```

O Prisma Studio estará disponível em `http://localhost:5555`.

A API estará disponível em `http://localhost:8000`.

A documentação da API estará disponível em `http://localhost:8000/docs`.

## Funcionalidades

- Criação e gerenciamento de reuniões
- Upload e transcrição de áudios de reuniões
- Geração automática de resumos de reuniões
- Extração de tópicos principais das reuniões

## Estrutura do Projeto

```
smart-meeting-api/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── meetings.py
│   │   │   └── transcriptions.py
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── events.py
│   ├── db/
│   │   ├── client.py
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── meeting.py
│   │   └── transcription.py
│   ├── services/
│   │   ├── meeting_service.py
│   │   └── transcription_service.py
│   ├── utils/
│   ├── __init__.py
│   └── main.py
├── prisma/
│   └── schema.prisma
├── scripts/
│   └── init_db.py
├── tests/
├── docker-compose.yml
├── Dockerfile
├── .env
├── Makefile
├── pyproject.toml
└── README.md
```

## Comandos Make

- `make install`: Instala as dependências
- `make run`: Executa a aplicação localmente
- `make test`: Executa os testes
- `make format`: Formata o código
- `make lint`: Executa o linter
- `make docker-up`: Inicia o contêiner do banco de dados
- `make docker-down`: Para todos os contêineres
- `make docker-build`: Constrói a imagem Docker da aplicação
- `make docker-run`: Executa a aplicação em Docker
- `make docker-all`: Executa todos os serviços em Docker
- `make prisma-init`: Inicializa o banco de dados
- `make prisma-generate`: Gera o cliente Prisma
- `make prisma-migrate`: Executa migrações do banco de dados
- `make prisma-studio`: Abre o Prisma Studio para visualizar/editar o banco de dados
- `make setup`: Executa o setup inicial completo 