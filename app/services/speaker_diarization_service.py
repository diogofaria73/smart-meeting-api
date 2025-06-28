"""
🎙️ SPEAKER DIARIZATION SERVICE - IDENTIFICAÇÃO DE PARTICIPANTES
Integra pyannote.audio com Whisper para identificar quem está falando
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
import time
import tempfile
import os
from pathlib import Path
import warnings

# Silencia warnings específicos do pyannote
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=FutureWarning, module="pyannote")

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    import torchaudio
    PYANNOTE_AVAILABLE = True
    logger.info("✅ pyannote.audio disponível")
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("⚠️ pyannote.audio não disponível. Instale com: pip install pyannote.audio")

from app.schemas.transcription import SpeakerSegment
from app.services.progress_service import progress_service


class SpeakerDiarizationService:
    """
    🎙️ SERVIÇO DE DIARIZAÇÃO DE SPEAKERS COM PYANNOTE.AUDIO
    
    Funcionalidades:
    - Identifica quantos speakers estão presentes
    - Segmenta o áudio por speaker
    - Integra com transcrição do Whisper
    - Otimizado para reuniões corporativas
    """
    
    def __init__(self):
        logger.info("🎙️ Inicializando SpeakerDiarizationService")
        self._pipeline = None
        self._device = self._detect_device()
        
        # Configurações otimizadas para reuniões
        self.min_speakers = 1
        self.max_speakers = 10  # Máximo razoável para reuniões
        self.min_segment_duration = 1.0  # Mínimo 1 segundo por segmento
        
        logger.info(f"🖥️ Device: {self._device}")
    
    def _detect_device(self) -> str:
        """Detecta o melhor device disponível"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    @property
    def pipeline(self) -> "Pipeline":
        """Carrega o pipeline de diarização com lazy loading"""
        if not PYANNOTE_AVAILABLE:
            raise RuntimeError("pyannote.audio não está disponível. Instale com: pip install pyannote.audio")
        
        if self._pipeline is None:
            logger.info("🔄 Carregando pipeline de speaker diarization...")
            start_time = time.time()
            
            try:
                # Usa o modelo pré-treinado mais recente
                # Carrega token do HuggingFace para acessar modelo gated
                from app.core.config import settings
                auth_token = settings.HUGGINGFACE_TOKEN
                
                if not auth_token or auth_token == "your_token_here":
                    logger.warning("⚠️ Token HuggingFace não configurado!")
                    logger.info("🔑 Para configurar:")
                    logger.info("   1. Obtenha token em: https://huggingface.co/settings/tokens")
                    logger.info("   2. Aceite termos em: https://huggingface.co/pyannote/speaker-diarization-3.1")
                    logger.info("   3. Configure HUGGINGFACE_TOKEN no arquivo .env.diarization")
                    
                logger.info(f"🔑 Usando token HuggingFace: {auth_token[:10]}..." if auth_token else "Sem token")
                    
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=auth_token if auth_token and auth_token != "your_token_here" else None
                )
                
                # Move para o device apropriado
                if self._device != "cpu":
                    self._pipeline = self._pipeline.to(torch.device(self._device))
                
                load_time = time.time() - start_time
                logger.info(f"✅ Pipeline de diarização carregado em {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar pipeline: {e}")
                logger.info("💡 Dica: Você pode precisar:")
                logger.info("   1. Aceitar os termos de uso no HuggingFace Hub")
                logger.info("   2. Configurar um token de autenticação")
                logger.info("   3. pip install pyannote.audio[pipeline]")
                raise RuntimeError(f"Falha ao carregar pipeline de diarização: {e}")
        
        return self._pipeline
    
    def diarize_audio(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🎯 EXECUTA DIARIZAÇÃO DO ÁUDIO
        
        Returns:
            Dict com speakers identificados e segmentos temporais
        """
        try:
            logger.info("🎙️ Iniciando diarização de speakers...")
            start_time = time.time()
            
            if task_id:
                progress_service.update_progress(task_id, "speaker_diarization", 10)
            
            # Prepara áudio para pyannote
            audio_tensor = self._prepare_audio_for_pyannote(audio_data, sample_rate)
            
            if task_id:
                progress_service.update_progress(task_id, "speaker_diarization", 30)
            
            # Executa diarização
            logger.info("🔄 Executando pipeline de diarização...")
            pipeline = self.pipeline
            
            # Aplica diarização
            diarization = pipeline({
                "waveform": audio_tensor.unsqueeze(0),
                "sample_rate": sample_rate
            })
            
            if task_id:
                progress_service.update_progress(task_id, "speaker_diarization", 70)
            
            # Processa resultados
            speakers_info = self._process_diarization_results(diarization)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Diarização concluída em {processing_time:.2f}s")
            logger.info(f"🎤 Speakers identificados: {speakers_info['speakers_count']}")
            
            if task_id:
                progress_service.update_progress(task_id, "speaker_diarization", 100)
            
            return {
                **speakers_info,
                "processing_time": processing_time,
                "audio_duration": len(audio_data) / sample_rate
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na diarização: {e}")
            if task_id:
                progress_service.update_progress(task_id, "speaker_diarization_error", 0)
            
            # Fallback: retorna speaker único
            return self._create_fallback_single_speaker(audio_data, sample_rate)
    
    def _prepare_audio_for_pyannote(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Prepara áudio para o pyannote.audio"""
        
        # Converte para tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.clone().detach().float()
        
        # Garante que está no formato correto (1D)
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)  # Converte para mono
        
        # Normaliza amplitude
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
        
        # Reamostra se necessário (pyannote espera 16kHz)
        if sample_rate != 16000:
            logger.info(f"🔄 Reamostrando de {sample_rate}Hz para 16kHz")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0), 
                sample_rate, 
                16000
            ).squeeze(0)
        
        return audio_tensor
    
    def _process_diarization_results(self, diarization: Annotation) -> Dict[str, Any]:
        """Processa os resultados da diarização"""
        
        segments = []
        speakers_stats = {}
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segment = {
                "start_time": float(segment.start),
                "end_time": float(segment.end),
                "duration": float(segment.duration),
                "speaker_id": str(speaker)
            }
            
            segments.append(speaker_segment)
            
            # Estatísticas por speaker
            if speaker not in speakers_stats:
                speakers_stats[speaker] = {
                    "total_time": 0.0,
                    "segments_count": 0,
                    "avg_segment_duration": 0.0
                }
            
            speakers_stats[speaker]["total_time"] += segment.duration
            speakers_stats[speaker]["segments_count"] += 1
        
        # Calcula médias
        for speaker, stats in speakers_stats.items():
            if stats["segments_count"] > 0:
                stats["avg_segment_duration"] = stats["total_time"] / stats["segments_count"]
        
        # Ordena segmentos por tempo
        segments.sort(key=lambda x: x["start_time"])
        
        return {
            "segments": segments,
            "speakers_count": len(speakers_stats),
            "speakers_stats": speakers_stats,
            "total_segments": len(segments)
        }
    
    def _create_fallback_single_speaker(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Cria resultado fallback com speaker único quando a diarização falha"""
        logger.warning("⚠️ Usando fallback: speaker único")
        
        duration = len(audio_data) / sample_rate
        
        return {
            "segments": [{
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration,
                "speaker_id": "SPEAKER_00"
            }],
            "speakers_count": 1,
            "speakers_stats": {
                "SPEAKER_00": {
                    "total_time": duration,
                    "segments_count": 1,
                    "avg_segment_duration": duration
                }
            },
            "total_segments": 1,
            "processing_time": 0.0,
            "audio_duration": duration,
            "fallback": True
        }
    
    def align_transcription_with_speakers(
        self,
        transcription_segments: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]]
    ) -> List[SpeakerSegment]:
        """
        🎯 ALINHA TRANSCRIÇÃO COM SPEAKERS
        
        Combina os segmentos de transcrição do Whisper com os speakers do pyannote
        """
        try:
            logger.info("🔄 Alinhando transcrição com speakers...")
            
            aligned_segments = []
            
            for trans_seg in transcription_segments:
                trans_start = trans_seg.get("start", 0.0)
                trans_end = trans_seg.get("end", trans_start + 1.0)
                trans_text = trans_seg.get("text", "").strip()
                
                if not trans_text:
                    continue
                
                # Encontra o speaker que mais se sobrepõe com este segmento
                best_speaker = self._find_best_speaker_for_segment(
                    trans_start, trans_end, speaker_segments
                )
                
                aligned_segment = SpeakerSegment(
                    start_time=trans_start,
                    end_time=trans_end,
                    speaker_id=best_speaker["speaker_id"],
                    text=trans_text,
                    confidence=best_speaker["confidence"]
                )
                
                aligned_segments.append(aligned_segment)
            
            logger.info(f"✅ Alinhamento concluído: {len(aligned_segments)} segmentos")
            return aligned_segments
            
        except Exception as e:
            logger.error(f"❌ Erro no alinhamento: {e}")
            return []
    
    def _find_best_speaker_for_segment(
        self, 
        start: float, 
        end: float, 
        speaker_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Encontra o speaker com maior sobreposição temporal"""
        
        best_overlap = 0.0
        best_speaker = {"speaker_id": "SPEAKER_00", "confidence": 0.5}
        
        segment_duration = end - start
        
        for speaker_seg in speaker_segments:
            # Calcula sobreposição
            overlap_start = max(start, speaker_seg["start_time"])
            overlap_end = min(end, speaker_seg["end_time"])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Percentual de sobreposição
            overlap_ratio = overlap_duration / segment_duration if segment_duration > 0 else 0
            
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = {
                    "speaker_id": speaker_seg["speaker_id"],
                    "confidence": min(0.95, 0.6 + overlap_ratio * 0.4)  # Confidence baseada na sobreposição
                }
        
        return best_speaker
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Retorna informações de performance do serviço"""
        return {
            "pyannote_available": PYANNOTE_AVAILABLE,
            "device": self._device,
            "pipeline_loaded": self._pipeline is not None,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "min_segment_duration": self.min_segment_duration
        }


# Instância global do serviço
speaker_diarization_service = SpeakerDiarizationService() 