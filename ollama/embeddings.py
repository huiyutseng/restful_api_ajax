# 取得文本的嵌入向量

import asyncio
from ollama import AsyncClient
import time


async def embed_text():
    t1 = time.time()
    response = await AsyncClient(
        host='http://localhost:11434',
        timeout=600,
    ).embed(
        model='bge-m3',
        input='The sky is blue because of Rayleigh scattering',
        keep_alive='1h',
    )
    print(response.embeddings)
    t2 = time.time()
    print(f'Response time: {t2 - t1:.2f} seconds')


asyncio.run(embed_text())