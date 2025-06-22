"""
🚀 FASTER WHISPER SERVICE - OTIMIZADO PARA MÁXIMA VELOCIDADE
Implementa transcrição com faster-whisper (até 4x mais rápido)
"""

import logging
import numpy as np
import torch
from typing import Optional, Dict, Any, List
import time
import psutil
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("⚠️ faster-whisper não disponível. Instale com: pip install faster-whisper")

from app.services.progress_service import progress_service


class FasterWhisperService:
    """
    ✅ SERVIÇO OTIMIZADO DE TRANSCRIÇÃO COM FASTER-WHISPER
    
    Benefícios:
    - 4x mais rápido que whisper original
    - Menor uso de memória
    - Suporte a quantização int8
    - Processamento em lote otimizado
    """
    
    def __init__(self):
        logger.info("🚀 Inicializando FasterWhisperService ULTRA OTIMIZADO")
        self._model = None
        self._device = self._detect_optimal_device()
        self._compute_type = self._detect_optimal_compute_type()
        self._model_size = self._select_optimal_model_size()
        
        logger.info(f"🖥️ Device: {self._device}")
        logger.info(f"⚙️ Compute type: {self._compute_type}")
        logger.info(f"📦 Model size: {self._model_size}")
    
    def _detect_optimal_device(self) -> str:
        """Detecta o melhor device para usar"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory >= 4.0:
                return "cuda"
            else:
                logger.info("⚠️ GPU com pouca memória, usando CPU")
                return "cpu"
        else:
            return "cpu"
    
    def _detect_optimal_compute_type(self) -> str:
        """Detecta o melhor tipo de computação"""
        if self._device == "cuda":
            # GPU: usar float16 para velocidade
            return "float16"
        else:
            # CPU: usar int8 para eficiência máxima
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / 1024**3
            
            logger.info(f"💻 CPU cores: {cpu_count}, RAM: {memory_gb:.1f}GB")
            
            if memory_gb >= 8.0:
                return "int8"  # Quantização para velocidade
            else:
                return "float32"  # Fallback para compatibilidade
    
    def _select_optimal_model_size(self) -> str:
        """Seleciona o tamanho ótimo do modelo baseado no hardware"""
        if self._device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 8.0:
                return "large-v3"    # GPU potente
            elif gpu_memory >= 4.0:
                return "medium"      # GPU média
            else:
                return "small"       # GPU limitada
        else:
            memory_gb = psutil.virtual_memory().total / 1024**3
            cpu_count = psutil.cpu_count()
            
            if memory_gb >= 16.0 and cpu_count >= 8:
                return "medium"      # CPU potente
            elif memory_gb >= 8.0:
                return "small"       # CPU média
            else:
                return "base"        # CPU limitada
    
    @property
    def model(self) -> "WhisperModel":
        """Carrega o modelo faster-whisper com lazy loading"""
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper não está disponível. Instale com: pip install faster-whisper")
        
        if self._model is None:
            logger.info(f"🔄 Carregando faster-whisper modelo: {self._model_size}")
            start_time = time.time()
            
            try:
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                    cpu_threads=min(8, psutil.cpu_count()) if self._device == "cpu" else 0,
                    num_workers=1  # Otimizado para single audio
                )
                
                load_time = time.time() - start_time
                logger.info(f"✅ Faster-whisper carregado em {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar faster-whisper: {e}")
                # Fallback para modelo menor
                logger.info("🔄 Tentando modelo menor como fallback...")
                self._model_size = "base"
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type="float32"  # Mais compatível
                )
                logger.info("✅ Faster-whisper carregado com modelo base")
        
        return self._model
    
    def transcribe_audio_optimized(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        task_id: Optional[str] = None
    ) -> str:
        """
        ✅ TRANSCRIÇÃO ULTRA OTIMIZADA COM FASTER-WHISPER
        """
        try:
            logger.info("⚡ Iniciando transcrição ULTRA OTIMIZADA com faster-whisper")
            start_time = time.time()
            
            # Atualiza progresso
            if task_id:
                progress_service.update_progress(task_id, "audio_processing", 60)
            
            # Normaliza áudio para faster-whisper
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normaliza amplitude
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            duration = len(audio_data) / sample_rate
            logger.info(f"🔊 Áudio processado: {duration:.2f}s, {sample_rate}Hz")
            
            # Configurações otimizadas para faster-whisper
            transcribe_options = {
                "language": "pt",
                "task": "transcribe",
                "beam_size": 1 if duration <= 30 else 2,  # Adaptativo
                "best_of": 1,  # Greedy para velocidade máxima
                "temperature": 0.0,  # Determinístico
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": False,  # Mais rápido
                "word_timestamps": False,  # Desabilita para velocidade
                "vad_filter": True,  # Voice Activity Detection
                "vad_parameters": {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": 30,
                    "min_silence_duration_ms": 100
                }
            }
            
            logger.info(f"🎯 Configurações faster-whisper: beam_size={transcribe_options['beam_size']}")
            
            # Atualiza progresso
            if task_id:
                progress_service.update_progress(task_id, "transcription_processing", 80)
            
            # Executa transcrição
            model = self.model
            segments, info = model.transcribe(
                audio_data,
                **transcribe_options
            )
            
            # Processa segmentos
            transcription_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    transcription_parts.append(text)
            
            # Junta resultado
            if transcription_parts:
                transcription_text = " ".join(transcription_parts)
            else:
                transcription_text = "[Áudio sem fala detectada]"
            
            # Pós-processamento PT-BR
            transcription_text = self._postprocess_portuguese_text(transcription_text)
            
            # Estatísticas de performance
            total_time = time.time() - start_time
            speed_ratio = duration / total_time if total_time > 0 else 0
            
            logger.info(f"✅ Faster-whisper CONCLUÍDO:")
            logger.info(f"   📊 Tempo: {total_time:.2f}s")
            logger.info(f"   🚀 Velocidade: {speed_ratio:.1f}x tempo real")
            logger.info(f"   📝 Caracteres: {len(transcription_text)}")
            logger.info(f"   🎯 Modelo: {self._model_size} ({self._compute_type})")
            
            # Atualiza progresso final
            if task_id:
                progress_service.update_progress(task_id, "transcription_processing", 100)
            
            return transcription_text
            
        except Exception as e:
            logger.error(f"❌ Erro na transcrição faster-whisper: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Erro na transcrição faster-whisper: {str(e)}")
    
    def _postprocess_portuguese_text(self, text: str) -> str:
        """Pós-processamento específico para português brasileiro"""
        if not text or text.strip() == "":
            return ""
        
        # Remove prefixos comuns do Whisper
        prefixes_to_remove = [
            "Transcrição:",
            "Texto:",
            "Audio:",
            "Áudio:",
            "[MÚSICA]",
            "[MUSIC]",
            "[Música]"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Capitaliza primeira letra
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Remove espaços múltiplos
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do sistema"""
        stats = {
            "device": self._device,
            "compute_type": self._compute_type,
            "model_size": self._model_size,
            "faster_whisper_available": FASTER_WHISPER_AVAILABLE
        }
        
        if self._device == "cuda" and torch.cuda.is_available():
            stats.update({
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_name": torch.cuda.get_device_name(0)
            })
        
        stats.update({
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / 1024**3,
            "memory_available": psutil.virtual_memory().available / 1024**3
        })
        
        return stats


# Instância global do serviço
faster_whisper_service = FasterWhisperService() 