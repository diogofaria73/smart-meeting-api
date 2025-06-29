# 🚀 Guia Rápido de Setup - Smart Meeting API

Este guia te ajudará a configurar o **Smart Meeting API** em sua máquina pela primeira vez.

## ⚡ Setup em 1 Comando (Recomendado)

Para configurar **tudo** automaticamente:

```bash
cd smart-meeting-api
make setup-first-time
```

✨ **Isso vai configurar TUDO automaticamente:**
- Verificar se Docker, Poetry e Python estão instalados
- Criar arquivos de configuração (.env)
- Subir PostgreSQL no Docker
- Instalar dependências Python
- Configurar banco de dados
- Baixar modelos de IA
- Instalar diarização de speakers

## 🔧 Requisitos do Sistema

Certifique-se de ter instalado:

- **Docker** e **Docker Compose**
- **Poetry** (gerenciador de dependências Python)
- **Python 3.10+**
- **Make** (opcional, mas recomendado)

### Instalação dos Requisitos

#### macOS (com Homebrew):
```bash
# Docker Desktop
brew install --cask docker

# Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Python (se necessário)
brew install python@3.11
```

#### Ubuntu/Debian:
```bash
# Docker
sudo apt update
sudo apt install docker.io docker-compose

# Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Python (se necessário)
sudo apt install python3.11 python3.11-pip
```

#### Windows:
- Instale [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Instale [Poetry](https://python-poetry.org/docs/#installation)
- Instale [Python 3.11+](https://www.python.org/downloads/)

## 🚀 Iniciando a Aplicação

Após o setup completo:

```bash
make run
```

A API estará disponível em: **http://localhost:8000**

📖 Documentação da API: **http://localhost:8000/docs**

## 🎯 Comandos Úteis

```bash
# Ver todos os comandos disponíveis
make help

# Parar a aplicação
Ctrl + C

# Parar Docker
make docker-down

# Interface gráfica do banco de dados
make prisma-studio

# Testar modelos de IA
make test-ai
```

## 🔍 Verificação de Problemas

Se algo der errado, você pode:

1. **Verificar se tudo está funcionando:**
   ```bash
   make check-performance
   ```

2. **Ver logs do Docker:**
   ```bash
   docker logs smart-meeting-db
   ```

3. **Reinstalar dependências:**
   ```bash
   make install
   ```

## 📁 Arquivos de Configuração

O setup criará automaticamente:

- **`.env`** - Configurações principais da API
- **`.env.diarization`** - Configurações para identificação de speakers

Você pode editar esses arquivos para personalizar a aplicação.

## 🎉 Próximos Passos

Após o setup completo, você pode:

1. **Testar a API** com um arquivo de áudio
2. **Explorar a documentação** em `/docs`
3. **Personalizar configurações** nos arquivos `.env`
4. **Configurar token do HuggingFace** para melhor diarização

---

**💡 Dica:** Se você é desenvolvedor, recomendamos usar o `make prisma-studio` para explorar a estrutura do banco de dados! 