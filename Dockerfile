FROM python:3.10-slim

WORKDIR /app

# Instala as dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala o Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Configura o Poetry para não criar ambiente virtual
RUN poetry config virtualenvs.create false

# Copia os arquivos de configuração do Poetry
COPY pyproject.toml poetry.lock* ./

# Instala as dependências
RUN poetry install --no-dev --no-interaction --no-ansi

# Copia o código da aplicação
COPY . .

# Gera o cliente Prisma
RUN poetry run prisma generate

# Expõe a porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 