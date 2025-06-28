#!/usr/bin/env python3
"""
🔍 DEBUG DO FLUXO DE TRANSCRIÇÃO
Identifica por que a diarização não está sendo executada
"""

import asyncio
import numpy as np
from pathlib import Path

async def debug_transcription_flow():
    print("🔍 DEBUG DO FLUXO DE TRANSCRIÇÃO")
    print("=" * 60)
    
    try:
        # 1. Verificar configurações
        from app.core.config import settings
        print(f"✅ CONFIGURAÇÕES:")
        print(f"   - ENABLE_SPEAKER_DIARIZATION: {settings.ENABLE_SPEAKER_DIARIZATION}")
        print(f"   - HUGGINGFACE_TOKEN: {'✅ Configurado' if settings.HUGGINGFACE_TOKEN else '❌ Não configurado'}")
        print(f"   - TRANSCRIPTION_MODEL: {settings.TRANSCRIPTION_MODEL}")
        
        # 2. Verificar disponibilidade dos serviços
        from app.services.transcription_service import TranscriptionService, ENHANCED_TRANSCRIPTION_AVAILABLE
        print(f"\n✅ SERVIÇOS:")
        print(f"   - ENHANCED_TRANSCRIPTION_AVAILABLE: {ENHANCED_TRANSCRIPTION_AVAILABLE}")
        
        # 3. Testar condições de hardware
        transcription_service = TranscriptionService()
        
        # Simular duração de áudio típica
        test_durations = [30.0, 60.0, 120.0, 300.0]  # 30s, 1min, 2min, 5min
        
        for duration in test_durations:
            should_use_enhanced = transcription_service._should_use_enhanced_transcription(duration)
            print(f"   - Duração {duration}s → Enhanced: {should_use_enhanced}")
        
        # 4. Verificar enhanced_transcription_service
        try:
            from app.services.enhanced_transcription_service import enhanced_transcription_service
            print(f"   - enhanced_transcription_service: ✅ Disponível")
            print(f"   - enable_diarization: {enhanced_transcription_service.enable_diarization}")
        except Exception as e:
            print(f"   - enhanced_transcription_service: ❌ {e}")
        
        # 5. Verificar speaker_diarization_service
        try:
            from app.services.speaker_diarization_service import speaker_diarization_service, PYANNOTE_AVAILABLE
            print(f"   - PYANNOTE_AVAILABLE: {PYANNOTE_AVAILABLE}")
            if PYANNOTE_AVAILABLE:
                # Testar carregamento do pipeline
                pipeline = speaker_diarization_service.pipeline
                print(f"   - Pipeline carregado: ✅")
            else:
                print(f"   - Pipeline: ❌ pyannote não disponível")
        except Exception as e:
            print(f"   - Pipeline: ❌ {e}")
        
        # 6. Simular áudio de teste para verificar fluxo
        print(f"\n🧪 SIMULANDO PROCESSAMENTO:")
        
        # Criar áudio fictício de 60 segundos
        sample_rate = 16000
        duration_seconds = 60.0
        audio_data = np.random.normal(0, 0.1, int(sample_rate * duration_seconds))
        
        print(f"   - Áudio simulado: {duration_seconds}s, {sample_rate}Hz")
        
        # Verificar se enhanced transcription seria usado
        should_use = transcription_service._should_use_enhanced_transcription(duration_seconds)
        print(f"   - Usaria Enhanced Transcription: {should_use}")
        
        if should_use and ENHANCED_TRANSCRIPTION_AVAILABLE:
            print(f"   - ✅ Diarização DEVERIA ser executada")
            
            # Testar enhanced transcription
            try:
                result = await enhanced_transcription_service.transcribe_with_speakers(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    enable_diarization=True
                )
                print(f"   - ✅ Enhanced transcription FUNCIONA")
                print(f"   - Speakers encontrados: {result.get('speakers_count', 0)}")
                print(f"   - Método: {result.get('method', 'N/A')}")
            except Exception as e:
                print(f"   - ❌ Enhanced transcription FALHOU: {e}")
        else:
            print(f"   - ❌ Diarização NÃO seria executada")
            print(f"     Motivos:")
            print(f"     - should_use_enhanced: {should_use}")
            print(f"     - ENHANCED_TRANSCRIPTION_AVAILABLE: {ENHANCED_TRANSCRIPTION_AVAILABLE}")
        
        print(f"\n🎯 CONCLUSÕES:")
        if should_use and ENHANCED_TRANSCRIPTION_AVAILABLE:
            print(f"   ✅ Sistema configurado corretamente para diarização")
            print(f"   💡 Se não está funcionando, problema pode estar em:")
            print(f"      1. Parâmetro enable_diarization=False no endpoint")
            print(f"      2. Fallback silencioso para transcrição simples")
            print(f"      3. Duração do áudio < 30s")
        else:
            print(f"   ❌ Sistema NÃO configurado para diarização")
        
    except Exception as e:
        print(f"❌ ERRO GERAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_transcription_flow())
