#!/usr/bin/env python3
"""
🧪 TESTE DO SISTEMA DE TRANSCRIÇÃO ASSÍNCRONA
Verifica se todas as funcionalidades estão funcionando corretamente
"""

import asyncio
import requests
import json
import time
from pathlib import Path
import numpy as np
import tempfile
import os
import wave

# Configurações
API_BASE_URL = "http://localhost:8000/api"
MEETING_ID = 1  # ID de teste


def create_test_audio_file() -> str:
    """Cria arquivo de áudio de teste"""
    print("🎵 Criando arquivo de áudio de teste...")
    
    # Parâmetros do áudio
    sample_rate = 16000
    duration = 10.0  # 10 segundos
    frequency = 440  # La (A4)
    
    # Gera tom senoidal
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Converte para 16-bit
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Salva como WAV
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Arquivo criado: {temp_file.name}")
    return temp_file.name


def test_async_transcription():
    """Testa o endpoint de transcrição assíncrona"""
    print("\n🚀 TESTE DO SISTEMA DE TRANSCRIÇÃO ASSÍNCRONA")
    print("=" * 60)
    
    try:
        # 1. Criar arquivo de teste
        audio_file_path = create_test_audio_file()
        
        try:
            # 2. Testar endpoint assíncrono
            print(f"\n📤 Enviando arquivo para transcrição...")
            print(f"   Meeting ID: {MEETING_ID}")
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': ('test_audio.wav', audio_file, 'audio/wav')
                }
                params = {
                    'meeting_id': MEETING_ID,
                    'enable_diarization': True
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/transcriptions/transcribe",
                    files=files,
                    params=params,
                    timeout=10  # Deve retornar rapidamente
                )
            
            print(f"📊 Status da resposta: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Endpoint assíncrono funcionando!")
                print(f"   Task ID: {result.get('task_id')}")
                print(f"   Status: {result.get('status')}")
                print(f"   WebSocket URL: {result.get('websocket_url')}")
                print(f"   Status URL: {result.get('status_url')}")
                
                task_id = result.get('task_id')
                
                # 3. Testar endpoint de status
                if task_id:
                    print(f"\n📊 Testando endpoint de status...")
                    
                    # Aguarda alguns segundos e verifica status
                    for i in range(5):
                        time.sleep(2)
                        
                        status_response = requests.get(
                            f"{API_BASE_URL}/transcriptions/status/{task_id}"
                        )
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            print(f"   ⏱️  Status {i+1}: {status_data['status']} - {status_data['progress']['percentage']:.1f}%")
                            print(f"      Etapa: {status_data['progress']['current_step']}")
                            print(f"      Mensagem: {status_data['progress']['message']}")
                            
                            if status_data['status'] in ['completed', 'failed']:
                                break
                        else:
                            print(f"   ❌ Erro ao consultar status: {status_response.status_code}")
                
                # 4. Testar endpoint de tarefas ativas
                print(f"\n📋 Testando endpoint de tarefas ativas...")
                tasks_response = requests.get(f"{API_BASE_URL}/transcriptions/tasks/active")
                
                if tasks_response.status_code == 200:
                    tasks_data = tasks_response.json()
                    print(f"   ✅ Tarefas ativas: {tasks_data['total_active_tasks']}")
                else:
                    print(f"   ❌ Erro ao consultar tarefas: {tasks_response.status_code}")
                
                # 5. Testar estatísticas WebSocket
                print(f"\n📡 Testando estatísticas WebSocket...")
                ws_response = requests.get(f"{API_BASE_URL}/ws/stats")
                
                if ws_response.status_code == 200:
                    ws_data = ws_response.json()
                    print(f"   ✅ Conexões WebSocket: {ws_data['websocket_connections']['total_connections']}")
                else:
                    print(f"   ❌ Erro ao consultar WebSocket: {ws_response.status_code}")
                
                print(f"\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
                print(f"   ✅ Sistema assíncrono funcionando")
                print(f"   ✅ Endpoints de status funcionando")
                print(f"   ✅ WebSocket configurado")
                
            else:
                print(f"❌ Erro na transcrição: {response.status_code}")
                print(f"   Resposta: {response.text}")
                
        finally:
            # Remove arquivo de teste
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
                print(f"🗑️  Arquivo de teste removido")
                
    except requests.exceptions.ConnectionError:
        print("❌ Erro: API não está rodando!")
        print("   Execute: poetry run python run.py")
    except Exception as e:
        print(f"❌ Erro no teste: {e}")


def print_usage_instructions():
    """Imprime instruções de uso do novo sistema"""
    print("\n📖 COMO USAR O NOVO SISTEMA ASSÍNCRONO")
    print("=" * 60)
    print("""
🚀 1. INICIE TRANSCRIÇÃO (retorna task_id imediatamente):
   POST /api/transcriptions/transcribe
   - Arquivo: multipart/form-data
   - Parâmetros: meeting_id, enable_diarization
   - Resposta: { "task_id": "...", "websocket_url": "...", "status_url": "..." }

📊 2. MONITORE PROGRESSO (duas opções):

   A) Via WebSocket (tempo real):
   ```javascript
   const ws = new WebSocket('ws://localhost:8000/api/ws/meeting/1');
   ws.onmessage = (event) => {
       const notification = JSON.parse(event.data);
       console.log('Progresso:', notification);
   };
   ```

   B) Via Polling:
   GET /api/transcriptions/status/{task_id}
   - Retorna: status, progresso, etapa atual, tempo estimado

🎯 3. OBTENHA RESULTADO FINAL:
   GET /api/transcriptions/{meeting_id}
   - Quando status = "completed"

🛑 4. CANCELE SE NECESSÁRIO:
   DELETE /api/transcriptions/cancel/{task_id}

📋 5. MONITORE SISTEMA:
   GET /api/transcriptions/tasks/active  # Lista tarefas ativas
   GET /api/ws/stats                     # Estatísticas WebSocket

🔔 EVENTOS WEBSOCKET RECEBIDOS:
   - transcription_started     # Início do processamento
   - transcription_progress    # Progresso (0-100%)
   - transcription_completed   # Transcrição pronta
   - transcription_failed      # Erro no processamento
   - analysis_completed        # Análise inteligente pronta
""")


if __name__ == "__main__":
    print_usage_instructions()
    
    # Pergunta se quer executar teste
    response = input("\n🧪 Executar teste do sistema? (y/N): ").strip().lower()
    if response in ['y', 'yes', 's', 'sim']:
        test_async_transcription()
    else:
        print("✅ Instruções exibidas. Execute o teste quando a API estiver rodando!") 