import asyncio
from contextlib import asynccontextmanager

from prisma import Prisma

# Cliente global do Prisma
prisma = Prisma()


async def init_prisma():
    """
    Inicializa a conexão com o banco de dados.
    """
    await prisma.connect()


async def close_prisma():
    """
    Fecha a conexão com o banco de dados.
    """
    if prisma.is_connected():
        await prisma.disconnect()


@asynccontextmanager
async def get_db():
    """
    Contexto para obter uma conexão com o banco de dados.
    """
    if not prisma.is_connected():
        await init_prisma()
    try:
        yield prisma
    except Exception as e:
        print(f"Erro na conexão com o banco de dados: {e}")
        raise 