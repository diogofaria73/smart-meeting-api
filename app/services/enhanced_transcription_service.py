"""
🚀 ENHANCED TRANSCRIPTION SERVICE - WHISPER + SPEAKER DIARIZATION
Combina transcrição do Whisper com identificação de speakers do pyannote.audio
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

from app.services.faster_whisper_service import FasterWhisperService
from app.services.speaker_diarization_service import speaker_diarization_service, PYANNOTE_AVAILABLE
from app.schemas.transcription import SpeakerSegment, ParticipantInfo
from app.services.progress_service import progress_service


class EnhancedTranscriptionService:
    """
    🚀 SERVIÇO APRIMORADO DE TRANSCRIÇÃO COM IDENTIFICAÇÃO DE SPEAKERS
    
    Funcionalidades:
    - Transcrição otimizada com Faster-Whisper
    - Identificação de speakers com pyannote.audio
    - Alinhamento inteligente entre texto e speakers
    - Estatísticas detalhadas de participação
    """
    
    def __init__(self):
        logger.info("🚀 Inicializando EnhancedTranscriptionService")
        self.whisper_service = FasterWhisperService()
        self.enable_diarization = PYANNOTE_AVAILABLE
        
        logger.info(f"🎙️ Diarização habilitada: {self.enable_diarization}")
    
    async def transcribe_with_speakers(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        task_id: Optional[str] = None,
        enable_diarization: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        🎯 TRANSCRIÇÃO COMPLETA COM IDENTIFICAÇÃO DE SPEAKERS
        
        Args:
            audio_data: Dados do áudio
            sample_rate: Taxa de amostragem
            task_id: ID da tarefa para tracking de progresso
            enable_diarization: Força habilitar/desabilitar diarização
            
        Returns:
            Dict com transcrição completa e speakers identificados
        """
        try:
            start_time = time.time()
            logger.info("🚀 Iniciando transcrição aprimorada com speakers...")
            
            # Determina se deve usar diarização
            use_diarization = (
                enable_diarization if enable_diarization is not None 
                else self.enable_diarization
            )
            
            if task_id:
                progress_service.update_progress(task_id, "initialization", 5)
            
            # Execução paralela quando possível
            if use_diarization:
                result = await self._transcribe_with_parallel_diarization(
                    audio_data, sample_rate, task_id
                )
            else:
                result = await self._transcribe_without_diarization(
                    audio_data, sample_rate, task_id
                )
            
            processing_time = time.time() - start_time
            result["total_processing_time"] = processing_time
            
            logger.info(f"✅ Transcrição aprimorada concluída em {processing_time:.2f}s")
            logger.info(f"🎤 Speakers identificados: {result.get('speakers_count', 1)}")
            
            if task_id:
                progress_service.update_progress(task_id, "completed", 100)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na transcrição aprimorada: {e}")
            if task_id:
                progress_service.update_progress(task_id, "error", 0)
            raise
    
    async def _transcribe_with_parallel_diarization(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executa transcrição e diarização em paralelo para máxima eficiência"""
        
        logger.info("⚡ Executando transcrição e diarização em paralelo...")
        
        # Cria tasks assíncronas para execução paralela
        transcription_task = asyncio.create_task(
            self._run_transcription_async(audio_data, sample_rate, task_id)
        )
        
        diarization_task = asyncio.create_task(
            self._run_diarization_async(audio_data, sample_rate, task_id)
        )
        
        # Aguarda ambos os processamentos
        transcription_result, diarization_result = await asyncio.gather(
            transcription_task, diarization_task
        )
        
        if task_id:
            progress_service.update_progress(task_id, "alignment", 90)
        
        # Alinha transcrição com speakers
        speaker_segments = speaker_diarization_service.align_transcription_with_speakers(
            transcription_result["segments"],
            diarization_result["segments"]
        )
        
        # Gera estatísticas de participação
        participants = self._generate_participant_stats(
            speaker_segments, diarization_result["speakers_stats"]
        )
        
        return {
            "transcription": transcription_result["text"],
            "speaker_segments": speaker_segments,
            "speakers_count": diarization_result["speakers_count"],
            "participants": participants,
            "processing_details": {
                "transcription_time": transcription_result["processing_time"],
                "diarization_time": diarization_result["processing_time"],
                "speakers_stats": diarization_result["speakers_stats"]
            },
            "confidence": self._calculate_overall_confidence(speaker_segments),
            "method": "whisper_plus_pyannote"
        }
    
    async def _transcribe_without_diarization(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executa apenas transcrição sem diarização"""
        
        logger.info("📝 Executando transcrição sem diarização...")
        
        transcription_result = await self._run_transcription_async(
            audio_data, sample_rate, task_id
        )
        
        # Cria um speaker único para todo o áudio
        duration = len(audio_data) / sample_rate
        single_segment = SpeakerSegment(
            start_time=0.0,
            end_time=duration,
            speaker_id="SPEAKER_00",
            text=transcription_result["text"],
            confidence=0.8
        )
        
        participants = [ParticipantInfo(
            name="Participante 1",
            speaker_id="SPEAKER_00",
            speaking_time=duration,
            segments_count=1,
            confidence=0.8
        )]
        
        return {
            "transcription": transcription_result["text"],
            "speaker_segments": [single_segment],
            "speakers_count": 1,
            "participants": participants,
            "processing_details": {
                "transcription_time": transcription_result["processing_time"],
                "diarization_time": 0.0
            },
            "confidence": 0.8,
            "method": "whisper_only"
        }
    
    async def _run_transcription_async(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executa transcrição de forma assíncrona"""
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Executa transcrição em thread separada
            transcription_text = await loop.run_in_executor(
                executor,
                self.whisper_service.transcribe_audio_optimized,
                audio_data,
                sample_rate,
                task_id
            )
            
            # Para obter segmentos detalhados, precisamos de uma versão que retorne timestamps
            segments = await loop.run_in_executor(
                executor,
                self._get_transcription_segments,
                audio_data,
                sample_rate
            )
        
        return {
            "text": transcription_text,
            "segments": segments,
            "processing_time": 0.0  # Will be calculated by the calling method
        }
    
    async def _run_diarization_async(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executa diarização de forma assíncrona"""
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                speaker_diarization_service.diarize_audio,
                audio_data,
                sample_rate,
                task_id
            )
        
        return result
    
    def _get_transcription_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Obtém segmentos de transcrição com timestamps"""
        
        try:
            # Garante que o áudio está no tipo correto (float32)
            if audio_data.dtype != np.float32:
                logger.info(f"🔄 Convertendo áudio de {audio_data.dtype} para float32")
                audio_data = audio_data.astype(np.float32)
            
            # Usa o modelo Whisper com timestamps habilitados
            model = self.whisper_service.model
            
            # Configurações para obter segmentos
            segments, _ = model.transcribe(
                audio_data,
                language="pt",
                word_timestamps=True,  # Habilita timestamps
                vad_filter=True
            )
            
            segment_list = []
            for segment in segments:
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
            
            return segment_list
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter segmentos: {e}")
            # Fallback: um segmento único
            duration = len(audio_data) / sample_rate
            return [{
                "start": 0.0,
                "end": duration,
                "text": "Transcrição completa"
            }]
    
    def _generate_participant_stats(
        self,
        speaker_segments: List[SpeakerSegment],
        speakers_stats: Dict[str, Any]
    ) -> List[ParticipantInfo]:
        """Gera estatísticas detalhadas dos participantes"""
        
        participants = []
        
        for speaker_id, stats in speakers_stats.items():
            # Calcula estatísticas de fala
            participant = ParticipantInfo(
                name=f"Participante {speaker_id.split('_')[-1]}",
                speaker_id=speaker_id,
                speaking_time=stats["total_time"],
                segments_count=stats["segments_count"],
                confidence=0.85,  # Base confidence para speakers identificados
                mentions=0  # Será calculado posteriormente pela análise de texto
            )
            
            participants.append(participant)
        
        # Ordena por tempo de fala (maior para menor)
        participants.sort(key=lambda p: p.speaking_time, reverse=True)
        
        return participants
    
    def _calculate_overall_confidence(
        self,
        speaker_segments: List[SpeakerSegment]
    ) -> float:
        """Calcula confiança geral baseada nos segmentos"""
        
        if not speaker_segments:
            return 0.0
        
        total_confidence = sum(seg.confidence for seg in speaker_segments)
        return min(0.95, total_confidence / len(speaker_segments))
    
    def get_service_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o serviço"""
        return {
            "whisper_available": True,
            "diarization_available": self.enable_diarization,
            "pyannote_info": speaker_diarization_service.get_performance_info(),
            "whisper_info": self.whisper_service.get_performance_stats()
        }


# Instância global do serviço aprimorado
enhanced_transcription_service = EnhancedTranscriptionService() 