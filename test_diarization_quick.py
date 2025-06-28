#!/usr/bin/env python3
"""
🚀 TESTE RÁPIDO DE DIARIZAÇÃO
Teste específico para verificar se o erro de tipo foi corrigido
"""

import asyncio
import numpy as np

async def test_diarization_quick():
    print("🚀 TESTE RÁPIDO DE DIARIZAÇÃO")
    print("=" * 40)
    
    try:
        from app.services.enhanced_transcription_service import enhanced_transcription_service
        
        # Criar áudio de teste com dtype explícito
        sample_rate = 16000
        duration_seconds = 30.0  # Áudio curto para teste rápido
        
        # Criar como float64 para testar conversão
        audio_data = np.random.normal(0, 0.1, int(sample_rate * duration_seconds)).astype(np.float64)
        
        print(f"✅ Áudio de teste criado:")
        print(f"   - Duração: {duration_seconds}s")
        print(f"   - Sample rate: {sample_rate}Hz")
        print(f"   - Dtype: {audio_data.dtype}")
        print(f"   - Shape: {audio_data.shape}")
        
        print(f"\n🎙️ Executando diarização...")
        
        result = await enhanced_transcription_service.transcribe_with_speakers(
            audio_data=audio_data,
            sample_rate=sample_rate,
            enable_diarization=True
        )
        
        print(f"\n✅ DIARIZAÇÃO CONCLUÍDA:")
        print(f"   - Speakers encontrados: {result.get('speakers_count', 0)}")
        print(f"   - Método: {result.get('method', 'N/A')}")
        print(f"   - Segmentos: {len(result.get('speaker_segments', []))}")
        print(f"   - Participantes: {len(result.get('participants', []))}")
        
        if result.get('speakers_count', 0) > 0:
            print(f"   🎯 SUCESSO: Diarização funcionando!")
        else:
            print(f"   ⚠️ 0 speakers - pode ser áudio sintético")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_diarization_quick())
