#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao PATH para importar módulos da aplicação
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.client import init_prisma, close_prisma


async def init_db():
    """
    Inicializa o banco de dados e cria as tabelas necessárias.
    """
    print("Inicializando banco de dados...")
    
    try:
        # Inicializa o cliente Prisma
        await init_prisma()
        print("Conexão com o banco de dados estabelecida com sucesso!")
        
        # Fecha a conexão
        await close_prisma()
        print("Banco de dados inicializado com sucesso!")
        
    except Exception as e:
        print(f"Erro ao inicializar o banco de dados: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(init_db()) 