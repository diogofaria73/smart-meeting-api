import json
import os
import tempfile
import logging
import psutil
from typing import List, Optional, Tuple
from pathlib import Path

import torch
import torchaudio
from fastapi import UploadFile, HTTPException
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    pipeline,
)
import numpy as np
from pydub import AudioSegment

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importação opcional do librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
    logger.info("✅ Librosa disponível")
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("⚠️ Librosa não está disponível. Usando apenas torchaudio e pydub.")

from app.core.config import settings
from app.db.client import get_db
from app.schemas.transcription import TranscriptionResponse, MeetingAnalysisResult
from app.services.progress_service import progress_service, ProgressStep

# ✅ OTIMIZAÇÃO: Importar faster-whisper se disponível
try:
    from app.services.faster_whisper_service import faster_whisper_service, FASTER_WHISPER_AVAILABLE
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    faster_whisper_service = None
    logger.warning("⚠️ FasterWhisperService não disponível")

# 🎙️ NOVA FUNCIONALIDADE: Importar serviço aprimorado com diarização
try:
    from app.services.enhanced_transcription_service import enhanced_transcription_service
    ENHANCED_TRANSCRIPTION_AVAILABLE = True
    logger.info("✅ Enhanced Transcription Service com diarização disponível")
except ImportError:
    ENHANCED_TRANSCRIPTION_AVAILABLE = False
    enhanced_transcription_service = None
    logger.warning("⚠️ Enhanced Transcription Service não disponível")

# 🧠 Importa o serviço de análise de IA
try:
    from app.services.meeting_analysis_service import meeting_analysis_service
    AI_ANALYSIS_AVAILABLE = True
    logger.info("✅ Serviço de IA disponível")
except ImportError:
    AI_ANALYSIS_AVAILABLE = False
    meeting_analysis_service = None
    logger.warning("⚠️ Serviço de IA não disponível, usando análise tradicional")


class TranscriptionService:
    """
    Serviço de transcrição para português brasileiro.
    
    Modelo único utilizado:
    - OpenAI Whisper Large-v3 (transcrição de áudio para texto PT-BR)
    - BERTimbau (sumarização de texto em português)
    """
    
    def __init__(self):
        logger.info("🚀 Inicializando TranscriptionService OTIMIZADO - Whisper + BERTimbau")
        # Inicialização preguiçosa dos modelos
        self._transcription_model = None
        self._summarization_model = None
        self._tokenizer = None
        self._processor = None
        self._model_name_used = "openai/whisper-large-v3"
        
        # ✅ OTIMIZAÇÕES: Detectar hardware disponível
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._gpu_memory_gb = self._get_gpu_memory()
        logger.info(f"🖥️ Hardware detectado: {self._device}")
        if self._device == "cuda":
            logger.info(f"💾 GPU Memory: {self._gpu_memory_gb:.1f}GB")
    
    def _get_gpu_memory(self) -> float:
        """Obtém memória GPU disponível em GB"""
        if torch.cuda.is_available():
            try:
                return torch.cuda.get_device_properties(0).total_memory / 1024**3
            except:
                return 0.0
        return 0.0
    
    def _get_optimal_config(self, duration_seconds: float) -> dict:
        """
        ✅ OTIMIZAÇÃO: Seleciona configuração ótima baseada na duração e hardware
        """
        logger.info(f"🎯 Selecionando configuração ótima para {duration_seconds:.1f}s de áudio")
        
        # Configurações baseadas na duração do áudio
        if duration_seconds <= 10:
            # Áudio muito curto - priorizar velocidade máxima
            config = {
                "max_new_tokens": 150,
                "num_beams": 1,          # Greedy decoding (mais rápido)
                "early_stopping": True,
                "chunk_duration": 10.0
            }
            logger.info("⚡ Configuração ULTRA RÁPIDA para áudio curto")
            
        elif duration_seconds <= 60:
            # Áudio médio - balanceado
            config = {
                "max_new_tokens": 200,
                "num_beams": 2,
                "early_stopping": True,
                "chunk_duration": 12.0
            }
            logger.info("🚀 Configuração RÁPIDA para áudio médio")
            
        elif self._device == "cuda" and self._gpu_memory_gb >= 6.0:
            # GPU com boa memória - pode usar configuração mais robusta
            config = {
                "max_new_tokens": 250,
                "num_beams": 2,
                "early_stopping": True,
                "chunk_duration": 15.0
            }
            logger.info("💪 Configuração BALANCEADA para GPU potente")
            
        else:
            # CPU ou GPU limitada - configuração conservadora
            config = {
                "max_new_tokens": 150,
                "num_beams": 1,
                "early_stopping": True,
                "chunk_duration": 10.0
            }
            logger.info("🔋 Configuração ECONÔMICA para hardware limitado")
        
        return config
    
    def _should_use_faster_whisper(self, duration_seconds: float) -> bool:
        """
        ✅ OTIMIZAÇÃO: Decide quando usar faster-whisper baseado na duração e hardware
        """
        if not FASTER_WHISPER_AVAILABLE:
            return False
        
        # Sempre usar faster-whisper para áudios curtos (mais rápido)
        if duration_seconds <= 60:
            logger.info(f"✅ Áudio curto ({duration_seconds:.1f}s): usando faster-whisper")
            return True
        
        # Para áudios longos, verificar hardware
        if self._device == "cuda" and self._gpu_memory_gb >= 4.0:
            logger.info(f"✅ GPU disponível: usando faster-whisper para {duration_seconds:.1f}s")
            return True
        
        # CPU com boa memória também pode usar
        memory_gb = psutil.virtual_memory().total / 1024**3
        if memory_gb >= 8.0:
            logger.info(f"✅ CPU com boa memória: usando faster-whisper para {duration_seconds:.1f}s")
            return True
        
        logger.info(f"⚠️ Hardware limitado: usando Whisper original para {duration_seconds:.1f}s")
        return False
    
    def _should_use_enhanced_transcription(self, duration_seconds: float) -> bool:
        """
        🎙️ NOVA FUNCIONALIDADE: Decide quando usar transcrição aprimorada com diarização
        """
        if not ENHANCED_TRANSCRIPTION_AVAILABLE:
            return False
        
        # Verificar se diarização está forçada
        from app.core.config import settings
        if settings.FORCE_DIARIZATION:
            logger.info(f"🔧 FORCE_DIARIZATION=True: usando transcrição aprimorada para {duration_seconds:.1f}s")
            return True
        
        # Usar transcrição aprimorada quando:
        # 1. Duração suficiente para ter múltiplos speakers (>30s)
        # 2. Hardware adequado disponível
        
        if duration_seconds < 30:
            logger.info(f"⏱️ Áudio muito curto ({duration_seconds:.1f}s): usando transcrição simples")
            return False
        
        # Verificar hardware disponível
        memory_gb = psutil.virtual_memory().total / 1024**3
        
        # Critérios mais permissivos para diarização
        if self._device == "cuda" and self._gpu_memory_gb >= 4.0:
            logger.info(f"🎙️ GPU CUDA disponível: usando transcrição aprimorada para {duration_seconds:.1f}s")
            return True
        elif self._device == "mps":  # Apple Silicon
            logger.info(f"🍎 Apple Silicon (MPS) detectado: usando transcrição aprimorada para {duration_seconds:.1f}s")
            return True
        elif memory_gb >= 6.0:  # Reduzido de 12GB para 6GB
            logger.info(f"🎙️ CPU com memória adequada ({memory_gb:.1f}GB): usando transcrição aprimorada para {duration_seconds:.1f}s")
            return True
        
        logger.info(f"⚠️ Hardware insuficiente para diarização (RAM: {memory_gb:.1f}GB): usando transcrição simples")
        logger.info(f"💡 Para forçar diarização, configure FORCE_DIARIZATION=true no .env")
        return False
    
    async def _transcribe_with_enhanced_service(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        task_id: Optional[str] = None
    ) -> dict:
        """
        🎙️ NOVA FUNCIONALIDADE: Transcrição aprimorada com identificação de speakers
        """
        try:
            logger.info("🎙️ Usando transcrição aprimorada com diarização...")
            
            if task_id:
                progress_service.update_progress(task_id, "enhanced_transcription_start", 20)
            
            # Chama o serviço aprimorado
            result = await enhanced_transcription_service.transcribe_with_speakers(
                audio_data=audio_data,
                sample_rate=sample_rate,
                task_id=task_id,
                enable_diarization=True
            )
            
            logger.info(f"✅ Transcrição aprimorada concluída:")
            logger.info(f"   - Speakers identificados: {result.get('speakers_count', 'N/A')}")
            logger.info(f"   - Método usado: {result.get('method', 'N/A')}")
            logger.info(f"   - Tempo total: {result.get('total_processing_time', 0):.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na transcrição aprimorada: {e}")
            logger.info("⚠️ Fazendo fallback para transcrição tradicional...")
            raise  # Re-raise para que o método principal possa fazer fallback
        
    @property
    def processor(self):
        """Property para acessar o processor de forma segura"""
        if self._processor is None:
            logger.info("🔄 Processor é None, carregando modelo primeiro")
            # Garante que o modelo seja carregado primeiro
            _ = self.transcription_model
        return self._processor
    
    @property
    def transcription_model(self):
        """Carrega APENAS o Whisper Large-v3 para transcrição em português brasileiro"""
        logger.info("🎯 Carregando Whisper Large-v3 para PT-BR")
        
        if self._transcription_model is None:
            logger.info("🔧 Inicializando Whisper Large-v3")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🖥️ Device detectado: {device}")
            
            model_name = settings.TRANSCRIPTION_MODEL
            logger.info(f"📋 Carregando modelo: {model_name}")
            
            try:
                logger.info(f"🎙️ Carregando Whisper Large-v3...")
                
                self._transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(device)
                
                self._processor = AutoProcessor.from_pretrained(model_name)
                
                # Configurar para português brasileiro
                if hasattr(self._processor, 'tokenizer'):
                    self._processor.tokenizer.set_prefix_tokens(language="portuguese")
                
                # Verificar se carregou corretamente
                if self._transcription_model is None or self._processor is None:
                    raise RuntimeError("Whisper Large-v3 não foi carregado corretamente")
                
                self._model_name_used = model_name
                logger.info(f"✅ SUCESSO: Whisper Large-v3 carregado!")
                logger.info(f"📊 Tipo do modelo: {type(self._transcription_model)}")
                logger.info(f"🎛️ Tipo do processor: {type(self._processor)}")
                logger.info(f"🖥️ Device do modelo: {next(self._transcription_model.parameters()).device}")
                
            except Exception as e:
                error_msg = f"❌ FALHA ao carregar Whisper Large-v3: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        return self._transcription_model
    
    @property
    def summarization_model(self):
        """Carrega modelo de sumarização otimizado para português brasileiro"""
        if self._summarization_model is None:
            logger.info("📝 Carregando modelo de sumarização PT-BR")
            
            try:
                # Usar BERTimbau para sumarização em português
                model_name = settings.SUMMARIZATION_MODEL
                logger.info(f"📚 Carregando BERTimbau: {model_name}")
                
                self._summarization_model = AutoModel.from_pretrained(model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                logger.info("✅ Modelo de sumarização PT-BR carregado com sucesso")
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar modelo de sumarização PT-BR: {e}")
                # Fallback para pipeline de sumarização
                try:
                    self._summarization_model = pipeline(
                        "summarization",
                        model="neuralmind/bert-base-portuguese-cased",
                        tokenizer="neuralmind/bert-base-portuguese-cased"
                    )
                    logger.info("✅ Pipeline de sumarização PT-BR carregado como fallback")
                except Exception as fallback_error:
                    logger.error(f"❌ Falha no fallback de sumarização: {fallback_error}")
                    raise
                
        return self._summarization_model
    
    def _convert_audio_to_wav(self, input_path: str, output_path: str) -> bool:
        """
        Converte áudio para formato WAV usando pydub como fallback
        """
        try:
            logger.info(f"Convertendo áudio {input_path} para WAV: {output_path}")
            # Tenta converter usando pydub (suporta mais formatos)
            audio = AudioSegment.from_file(input_path)
            # Converte para mono, 16kHz, 16-bit
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio.export(output_path, format="wav")
            logger.info("Conversão para WAV bem-sucedida")
            return True
        except Exception as e:
            logger.error(f"Erro ao converter áudio com pydub: {e}")
            return False
    
    def _load_audio_with_librosa(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Carrega áudio usando librosa (se disponível)
        """
        logger.info(f"Tentando carregar áudio com librosa: {file_path}")
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa não está disponível")
        
        audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        logger.info(f"Áudio carregado com librosa: shape={audio_data.shape}, sr={sample_rate}")
        return audio_data, sample_rate
    
    def _load_audio_with_torchaudio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Carrega áudio usando torchaudio
        """
        logger.info(f"Tentando carregar áudio com torchaudio: {file_path}")
        waveform, sample_rate = torchaudio.load(file_path)
        logger.info(f"Áudio carregado - shape original: {waveform.shape}, sr: {sample_rate}")
        
        # Converter para mono se necessário
        if waveform.shape[0] > 1:
            logger.info("Convertendo para mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample se necessário
        if sample_rate != 16000:
            logger.info(f"Fazendo resample de {sample_rate}Hz para 16000Hz")
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)
            sample_rate = 16000
        
        result = waveform.squeeze().numpy(), sample_rate
        logger.info(f"Áudio processado com torchaudio: shape={result[0].shape}, sr={result[1]}")
        return result
    
    def _resample_with_torchaudio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        """
        Faz resample usando torchaudio
        """
        waveform = torch.from_numpy(audio_data).unsqueeze(0)
        transform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = transform(waveform)
        return resampled.squeeze().numpy()
    
    def _load_audio_robust(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Carrega áudio de forma robusta, tentando diferentes métodos
        """
        logger.info(f"=== Iniciando carregamento robusto de áudio: {file_path} ===")
        
        # Método 1: Tentar com librosa (mais robusto) se disponível
        if LIBROSA_AVAILABLE:
            try:
                logger.info("MÉTODO 1: Tentando com librosa")
                audio_data, sample_rate = self._load_audio_with_librosa(file_path)
                logger.info("✅ SUCESSO com librosa")
                return audio_data, sample_rate
            except Exception as e:
                logger.error(f"❌ FALHOU com librosa: {e}")
        
        # Método 2: Tentar com torchaudio diretamente
        try:
            logger.info("MÉTODO 2: Tentando com torchaudio direto")
            audio_data, sample_rate = self._load_audio_with_torchaudio(file_path)
            logger.info("✅ SUCESSO com torchaudio direto")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"❌ FALHOU com torchaudio direto: {e}")
        
        # Método 3: Tentar converter para WAV primeiro
        try:
            logger.info("MÉTODO 3: Tentando conversão para WAV primeiro")
            temp_wav = file_path + "_converted.wav"
            if self._convert_audio_to_wav(file_path, temp_wav):
                try:
                    # Tentar com torchaudio no arquivo convertido
                    audio_data, sample_rate = self._load_audio_with_torchaudio(temp_wav)
                    os.unlink(temp_wav)  # Remove arquivo temporário
                    logger.info("✅ SUCESSO com conversão WAV + torchaudio")
                    return audio_data, sample_rate
                except Exception as e2:
                    logger.error(f"❌ FALHOU com torchaudio no arquivo convertido: {e2}")
                    if os.path.exists(temp_wav):
                        os.unlink(temp_wav)
        except Exception as e:
            logger.error(f"❌ FALHOU na conversão para WAV: {e}")
        
        # Método 4: Tentar usando pydub para extrair dados do áudio
        try:
            logger.info("MÉTODO 4: Tentando com pydub direto")
            audio = AudioSegment.from_file(file_path)
            # Converte para mono e 16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            # Converte para numpy array
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # Normaliza para -1 a 1
            audio_data = audio_data / (2**15)  # Para 16-bit
            logger.info(f"✅ SUCESSO com pydub direto: shape={audio_data.shape}")
            return audio_data, 16000
        except Exception as e:
            logger.error(f"❌ FALHOU com pydub: {e}")
        
        error_msg = "Não foi possível processar o arquivo de áudio com nenhum método"
        logger.error(f"❌ FALHA TOTAL no carregamento de áudio: {error_msg}")
        raise HTTPException(
            status_code=400, 
            detail="Não foi possível processar o arquivo de áudio. Formatos suportados: MP3, WAV, M4A, FLAC, OGG. "
                   "Verifique se o arquivo não está corrompido."
        )

    async def transcribe_audio(self, meeting_id: int, file: UploadFile, enable_diarization: bool = True, task_id: Optional[str] = None) -> TranscriptionResponse:
        """
        Transcreve um arquivo de áudio e salva no banco de dados.
        """
        logger.info(f"=== INICIANDO TRANSCRIÇÃO ===")
        logger.info(f"Meeting ID: {meeting_id}")
        logger.info(f"Arquivo: {file.filename}")
        logger.info(f"Content-Type: {file.content_type}")
        
        # Atualiza progresso - Upload e Validação
        if task_id:
            progress_service.update_progress(
                task_id, 
                ProgressStep.UPLOAD_VALIDATION,
                f"Validando arquivo: {file.filename}",
                details=f"Tipo: {file.content_type}"
            )
        
        # Validar tipo de arquivo
        allowed_types = [
            "audio/wav", "audio/mp3", "audio/mpeg", "audio/mp4", 
            "audio/m4a", "audio/flac", "audio/ogg", "audio/webm"
        ]
        
        if file.content_type and file.content_type not in allowed_types:
            error_msg = f"Tipo de arquivo não suportado: {file.content_type}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=f"{error_msg}. Tipos suportados: {', '.join(allowed_types)}"
            )
        
        # Detectar extensão do arquivo
        file_extension = ""
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            logger.info(f"Extensão detectada pelo filename: {file_extension}")
        elif file.content_type:
            extension_map = {
                "audio/wav": ".wav",
                "audio/mp3": ".mp3", 
                "audio/mpeg": ".mp3",
                "audio/mp4": ".m4a",
                "audio/m4a": ".m4a",
                "audio/flac": ".flac",
                "audio/ogg": ".ogg",
                "audio/webm": ".webm"
            }
            file_extension = extension_map.get(file.content_type, ".wav")
            logger.info(f"Extensão detectada pelo content-type: {file_extension}")
        
        # Salva o arquivo temporariamente com extensão correta
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            logger.info(f"Criando arquivo temporário: {temp_file.name}")
            content = await file.read()
            logger.info(f"Lidos {len(content)} bytes do arquivo")
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Arquivo salvo em: {temp_file_path}")
        
        try:
            # Atualiza progresso - Pré-processamento de áudio
            if task_id:
                progress_service.update_progress(
                    task_id,
                    ProgressStep.AUDIO_PREPROCESSING,
                    "Carregando e processando arquivo de áudio...",
                    details="Convertendo formato e normalizando áudio"
                )
            
            logger.info("=== INICIANDO CARREGAMENTO DE ÁUDIO ===")
            # Carrega o áudio de forma robusta
            audio_data, sample_rate = self._load_audio_robust(temp_file_path)
            logger.info(f"✅ Áudio carregado com sucesso: shape={audio_data.shape}, sr={sample_rate}")
            
            # Verifica se o áudio não está vazio
            if len(audio_data) == 0:
                error_msg = "Arquivo de áudio está vazio"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Verifica se o áudio tem duração mínima (0.1 segundos)
            min_duration = 0.1
            duration = len(audio_data) / sample_rate
            logger.info(f"Duração do áudio: {duration:.2f} segundos")
            
            if duration < min_duration:
                error_msg = f"Arquivo de áudio muito curto. Duração: {duration:.2f}s, mínima: {min_duration}s"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Atualiza progresso - Carregamento do modelo
            if task_id:
                progress_service.update_progress(
                    task_id,
                    ProgressStep.MODEL_LOADING,
                    "Carregando modelo Whisper Large-v3...",
                    details="Preparando modelo de IA para transcrição"
                )
            
            logger.info("=== INICIANDO GERAÇÃO DE TRANSCRIÇÃO ===")
            
            # Atualiza progresso - Início da transcrição
            if task_id:
                progress_service.update_progress(
                    task_id,
                    ProgressStep.TRANSCRIPTION,
                    f"Iniciando transcrição do áudio ({duration:.1f}s)...",
                    details=f"Duração: {duration:.1f} segundos"
                )
                # Atualiza informações de áudio
                progress_info = progress_service.get_progress(task_id)
                if progress_info:
                    progress_info.audio_duration_seconds = duration
            
            # ✅ NOVA OTIMIZAÇÃO: Escolher melhor engine de transcrição com diarização
            use_enhanced_transcription = ENHANCED_TRANSCRIPTION_AVAILABLE and enable_diarization and self._should_use_enhanced_transcription(duration)
            use_faster_whisper = FASTER_WHISPER_AVAILABLE and self._should_use_faster_whisper(duration)
            
            enhanced_result = None
            
            if use_enhanced_transcription:
                try:
                    logger.info("🎙️ Usando TRANSCRIÇÃO APRIMORADA com identificação de speakers")
                    enhanced_result = await self._transcribe_with_enhanced_service(
                        audio_data, sample_rate, task_id
                    )
                    transcription_text = enhanced_result["transcription"]
                    logger.info(f"✅ Transcrição aprimorada bem-sucedida: {enhanced_result.get('speakers_count', 1)} speakers")
                except Exception as e:
                    logger.error(f"❌ Erro na transcrição aprimorada: {e}")
                    logger.info("⚠️ Fazendo fallback para transcrição tradicional...")
                    enhanced_result = None
            
            # Fallback para engines tradicionais se necessário
            if enhanced_result is None:
                if use_faster_whisper:
                    logger.info("🚀 Usando FASTER-WHISPER para máxima velocidade")
                    transcription_text = faster_whisper_service.transcribe_audio_optimized(
                        audio_data, sample_rate, task_id
                    )
                else:
                    # Usar Whisper original com otimizações
                    optimal_config = self._get_optimal_config(duration)
                    max_duration = optimal_config["chunk_duration"]
                    
                    if duration > max_duration:
                        logger.info(f"Áudio longo detectado ({duration:.2f}s), segmentando em chunks de {max_duration}s")
                        transcription_text = self._transcribe_long_audio_optimized(audio_data, sample_rate, optimal_config, task_id)
                    else:
                        logger.info(f"Áudio curto ({duration:.2f}s), transcrição direta otimizada")
                        transcription_text = self._generate_transcription_optimized(audio_data, sample_rate, optimal_config)
            
            logger.info(f"✅ Transcrição gerada: {len(transcription_text)} caracteres")
            logger.info(f"Prévia da transcrição: {transcription_text[:200]}...")
            
            # Verifica se a transcrição não está vazia
            if not transcription_text.strip():
                transcription_text = "[Áudio sem fala detectada ou muito baixo]"
                logger.warning("Transcrição vazia, usando mensagem padrão")
            
            # Atualiza progresso - Pós-processamento
            if task_id:
                progress_service.update_progress(
                    task_id,
                    ProgressStep.POST_PROCESSING,
                    "Pós-processando texto transcrito...",
                    details="Aplicando correções e formatação"
                )
            
            logger.info("=== SALVANDO NO BANCO DE DADOS ===")
            
            # Atualiza progresso - Salvamento no banco
            if task_id:
                progress_service.update_progress(
                    task_id,
                    ProgressStep.DATABASE_SAVE,
                    "Salvando transcrição no banco de dados...",
                    details="Persistindo dados da transcrição"
                )
            # Salva a transcrição no banco de dados
            async with get_db() as db:
                # Verifica se a reunião existe
                logger.info(f"Verificando se reunião {meeting_id} existe")
                meeting = await db.meeting.find_unique(where={"id": meeting_id})
                if not meeting:
                    error_msg = f"Reunião com ID {meeting_id} não encontrada"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info("Reunião encontrada, criando transcrição")
                
                # Prepara dados para salvamento
                transcription_data = {
                    "meeting_id": meeting_id,
                    "content": transcription_text,
                }
                
                # 🎙️ NOVA FUNCIONALIDADE: Salva informações de speakers se disponível
                if enhanced_result:
                    logger.info(f"💾 Salvando dados aprimorados: {enhanced_result.get('speakers_count', 1)} speakers")
                    
                    # 🔧 CORREÇÃO: Converte objetos Pydantic para dicts antes da serialização JSON
                    import json
                    
                    # Converte speaker_segments para dicts se necessário
                    speaker_segments_for_db = []
                    for segment in enhanced_result.get('speaker_segments', []):
                        if hasattr(segment, 'dict'):  # É um objeto Pydantic
                            speaker_segments_for_db.append(segment.dict())
                        else:  # Já é um dict
                            speaker_segments_for_db.append(segment)
                    
                    # Converte participants para dicts se necessário  
                    participants_for_db = []
                    for participant in enhanced_result.get('participants', []):
                        if hasattr(participant, 'dict'):  # É um objeto Pydantic
                            participants_for_db.append(participant.dict())
                        else:  # Já é um dict
                            participants_for_db.append(participant)
                    
                    # Adiciona campos de diarização ao objeto de salvamento
                    transcription_data.update({
                        "speakers_count": enhanced_result.get('speakers_count', 0),
                        "speaker_segments": json.dumps(speaker_segments_for_db, ensure_ascii=False),
                        "participants": json.dumps(participants_for_db, ensure_ascii=False),
                        "diarization_method": enhanced_result.get('method', 'whisper_plus_pyannote'),
                        "processing_details": json.dumps({
                            "transcription_time": enhanced_result.get('transcription_time', 0),
                            "diarization_time": enhanced_result.get('diarization_time', 0),
                            "total_time": enhanced_result.get('total_processing_time', 0),
                            "confidence": enhanced_result.get('confidence', 0.8),
                            "audio_duration": duration
                        }, ensure_ascii=False)
                    })
                    
                    logger.info(f"   - Método usado: {enhanced_result.get('method', 'N/A')}")
                    logger.info(f"   - Confiança geral: {enhanced_result.get('confidence', 0):.2f}")
                    logger.info(f"   - Tempo de processamento: {enhanced_result.get('total_processing_time', 0):.2f}s")
                
                # Cria a transcrição
                transcription = await db.transcription.create(data=transcription_data)
                
                logger.info(f"Transcrição criada com ID: {transcription.id}")
                
                # Atualiza o status da reunião
                await db.meeting.update(
                    where={"id": meeting_id},
                    data={"has_transcription": True}
                )
                
                logger.info("Status da reunião atualizado")
                
                # Atualiza progresso - Conclusão
                if task_id:
                    progress_service.mark_completed(task_id)
                
                logger.info("=== TRANSCRIÇÃO CONCLUÍDA COM SUCESSO ===")
                
                # Prepara resposta com dados de diarização se disponível
                response_data = {
                    "id": transcription.id,
                    "meeting_id": transcription.meeting_id,
                    "content": transcription.content,
                    "created_at": transcription.created_at,
                    "updated_at": transcription.updated_at,
                    "is_summarized": transcription.is_summarized,
                    "is_analyzed": getattr(transcription, 'is_analyzed', False),
                    "summary": None,
                    "topics": [],
                    "analysis": None,
                    "speakers_count": transcription.speakers_count or 0,
                    "speaker_segments": [],
                    "participants": [], 
                    "diarization_method": transcription.diarization_method,
                    "processing_details": None
                }
                
                # Adiciona dados de diarização se disponível
                if enhanced_result:
                    import json
                    
                    # 🔧 CORREÇÃO: Converte objetos Pydantic para dicts antes da serialização
                    speaker_segments_dict = []
                    for segment in enhanced_result.get('speaker_segments', []):
                        if hasattr(segment, 'dict'):  # É um objeto Pydantic
                            speaker_segments_dict.append(segment.dict())
                        else:  # Já é um dict
                            speaker_segments_dict.append(segment)
                    
                    participants_dict = []
                    for participant in enhanced_result.get('participants', []):
                        if hasattr(participant, 'dict'):  # É um objeto Pydantic  
                            participants_dict.append(participant.dict())
                        else:  # Já é um dict
                            participants_dict.append(participant)
                    
                    response_data.update({
                        "speaker_segments": speaker_segments_dict,
                        "participants": participants_dict,
                        "processing_details": {
                            "transcription_time": enhanced_result.get('transcription_time', 0),
                            "diarization_time": enhanced_result.get('diarization_time', 0),
                            "total_time": enhanced_result.get('total_processing_time', 0),
                            "confidence": enhanced_result.get('confidence', 0.8),
                            "audio_duration": duration
                        }
                    })
                
                return TranscriptionResponse(**response_data)
        except HTTPException:
            logger.error("Re-raising HTTPException")
            raise
        except Exception as e:
            # Marca progresso como falhado
            if task_id:
                progress_service.mark_failed(task_id, str(e))
            
            logger.error(f"❌ ERRO GERAL na transcrição: {str(e)}")
            logger.error(f"Tipo do erro: {type(e)}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro interno na transcrição: {str(e)}"
            )
        finally:
            # Remove o arquivo temporário
            if os.path.exists(temp_file_path):
                logger.info(f"Removendo arquivo temporário: {temp_file_path}")
                os.unlink(temp_file_path)
            else:
                logger.warning(f"Arquivo temporário não encontrado para remoção: {temp_file_path}")

    async def generate_summary(self, meeting_id: int) -> TranscriptionResponse:
        """
        Gera um resumo INTELIGENTE com análise completa para português brasileiro.
        Extrai participantes, tópicos, tarefas e decisões automaticamente.
        """
        logger.info(f"📝 Gerando resumo e análise inteligente PT-BR para reunião {meeting_id}")
        
        async with get_db() as db:
            # Busca a transcrição
            transcription = await db.transcription.find_first(
                where={"meeting_id": meeting_id}
            )
            
            if not transcription:
                error_msg = f"Transcrição para reunião com ID {meeting_id} não encontrada"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Verifica se já foi analisada
            if transcription.is_summarized and transcription.is_analyzed:
                logger.info("Transcrição já possui análise completa, retornando dados existentes")
                existing_analysis = await self._get_existing_analysis(meeting_id)
                existing_summary = await self._get_existing_summary(meeting_id)
                existing_topics = await self._get_existing_topics(meeting_id)
                
                return TranscriptionResponse(
                    id=transcription.id,
                    meeting_id=transcription.meeting_id,
                    content=transcription.content,
                    created_at=transcription.created_at,
                    updated_at=transcription.updated_at,
                    is_summarized=transcription.is_summarized,
                    is_analyzed=transcription.is_analyzed,
                    summary=existing_summary,
                    topics=existing_topics,
                    analysis=existing_analysis
                )
            
            logger.info(f"📄 Transcrição encontrada: {len(transcription.content)} caracteres")
            
            try:
                # 🧠 ANÁLISE INTELIGENTE COM IA OU FALLBACK TRADICIONAL
                if AI_ANALYSIS_AVAILABLE and meeting_analysis_service:
                    logger.info("🤖 Iniciando análise com IA otimizada")
                    analysis_result = await meeting_analysis_service.analyze_meeting(
                        transcription.content
                    )
                    logger.info(f"✅ Análise IA concluída em {analysis_result.processing_time:.2f}s")
                else:
                    logger.info("🔍 Iniciando análise tradicional (fallback)")
                    analysis_result = await meeting_analysis_service.analyze_meeting(
                        transcription_text=transcription.content,
                        include_sentiment=True,
                        extract_participants=True,
                        extract_action_items=True,
                        min_confidence=0.6
                    )
                
                # Usa o resumo da análise inteligente ou gera um tradicional
                if analysis_result.summary and len(analysis_result.summary) > 50:
                    summary = analysis_result.summary
                    logger.info("✅ Usando resumo da análise inteligente")
                else:
                    logger.info("📝 Gerando resumo tradicional como fallback")
                    summary = await self._generate_portuguese_summary(transcription.content)
                
                # Extrai tópicos da análise inteligente ou método tradicional
                if analysis_result.main_topics:
                    topics = [topic.title for topic in analysis_result.main_topics]
                    logger.info(f"✅ Usando tópicos da análise inteligente: {len(topics)}")
                else:
                    logger.info("🏷️ Extraindo tópicos pelo método tradicional")
                    topics = self._extract_portuguese_topics(transcription.content)
                
                logger.info(f"📊 RESULTADOS DA ANÁLISE:")
                logger.info(f"   • Resumo: {len(summary)} caracteres")
                logger.info(f"   • Participantes: {len(analysis_result.participants)}")
                logger.info(f"   • Tópicos: {len(analysis_result.main_topics)}")
                logger.info(f"   • Itens de ação: {len(analysis_result.action_items)}")
                logger.info(f"   • Decisões: {len(analysis_result.key_decisions)}")
                logger.info(f"   • Confiança: {analysis_result.confidence_score:.2f}")
                
                # Salva o resumo tradicional
                await db.summary.create(
                    data={
                        "meeting_id": meeting_id,
                        "content": summary,
                        "topics": json.dumps(topics, ensure_ascii=False),
                    }
                )
                
                # 🆕 Salva a análise inteligente completa
                analysis_data = {
                    "meeting_id": meeting_id,
                    "participants": json.dumps([p.dict() for p in analysis_result.participants], ensure_ascii=False),
                    "main_topics": json.dumps([t.dict() for t in analysis_result.main_topics], ensure_ascii=False),
                    "action_items": json.dumps([a.dict() for a in analysis_result.action_items], ensure_ascii=False),
                    "key_decisions": json.dumps([d.dict() for d in analysis_result.key_decisions], ensure_ascii=False),
                    "summary": analysis_result.summary,
                    "confidence_score": analysis_result.confidence_score
                }
                
                # Adiciona análise de sentimento se disponível
                if analysis_result.sentiment_analysis:
                    analysis_data["sentiment_analysis"] = json.dumps(
                        analysis_result.sentiment_analysis.dict(), ensure_ascii=False
                    )
                
                await db.meetinganalysis.create(data=analysis_data)
                
                # Atualiza o status da transcrição e da reunião
                await db.transcription.update(
                    where={"id": transcription.id},
                    data={
                        "is_summarized": True,
                        "is_analyzed": True
                    }
                )
                
                await db.meeting.update(
                    where={"id": meeting_id},
                    data={
                        "has_summary": True,
                        "has_analysis": True
                    }
                )
                
                logger.info("✅ Resumo e análise inteligente salvos com sucesso")
                
                # Retorna a transcrição atualizada com análise completa
                updated_transcription = await db.transcription.find_unique(
                    where={"id": transcription.id}
                )
                
                return TranscriptionResponse(
                    id=updated_transcription.id,
                    meeting_id=updated_transcription.meeting_id,
                    content=updated_transcription.content,
                    created_at=updated_transcription.created_at,
                    updated_at=updated_transcription.updated_at,
                    is_summarized=updated_transcription.is_summarized,
                    is_analyzed=updated_transcription.is_analyzed,
                    summary=summary,
                    topics=topics,
                    analysis=analysis_result
                )
                
            except Exception as e:
                logger.error(f"❌ Erro ao gerar resumo e análise inteligente: {e}")
                raise RuntimeError(f"Falha na geração de resumo: {str(e)}")
    
    async def _generate_portuguese_summary(self, text: str) -> str:
        """
        Gera resumo otimizado para português brasileiro usando múltiplas estratégias.
        """
        logger.info("🎯 Iniciando geração de resumo PT-BR")
        
        try:
            # Estratégia 1: Usar pipeline de sumarização (mais estável)
            if hasattr(self.summarization_model, '__call__'):
                logger.info("📋 Usando pipeline de sumarização PT-BR")
                
                # Limita o texto para evitar problemas de memória
                max_input_length = 1000
                if len(text) > max_input_length:
                    text = text[:max_input_length] + "..."
                
                summary = self.summarization_model(
                    text,
                    max_length=200,
                    min_length=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                
                if isinstance(summary, list) and len(summary) > 0:
                    return summary[0].get('summary_text', summary[0].get('text', str(summary[0])))
                else:
                    return str(summary)
            
            # Estratégia 2: Sumarização baseada em sentenças (fallback)
            else:
                logger.info("📄 Usando sumarização baseada em sentenças PT-BR")
                return self._extractive_summary_portuguese(text)
                
        except Exception as e:
            logger.error(f"❌ Erro na sumarização PT-BR: {e}")
            # Fallback para resumo extrativo simples
            return self._extractive_summary_portuguese(text)
    
    def _extractive_summary_portuguese(self, text: str) -> str:
        """
        Gera resumo extrativo otimizado para estruturas de português brasileiro.
        """
        logger.info("🔍 Gerando resumo extrativo PT-BR")
        
        # Palavras-chave importantes em reuniões brasileiras
        keywords_pt = [
            'decidiu', 'definiu', 'acordou', 'resolveu', 'ficou definido',
            'conclusão', 'resultado', 'importante', 'principal', 'destacou',
            'discutiu', 'apresentou', 'propôs', 'sugeriu', 'recomendou',
            'próximos passos', 'ações', 'responsável', 'prazo', 'data'
        ]
        
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        # Pontua sentenças baseadas em palavras-chave PT-BR
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Aumenta score para sentenças com palavras-chave
            for keyword in keywords_pt:
                if keyword in sentence_lower:
                    score += 2
            
            # Penaliza sentenças muito curtas ou muito longas
            if len(sentence.split()) < 5:
                score -= 1
            elif len(sentence.split()) > 30:
                score -= 1
            
            sentence_scores.append((sentence, score))
        
        # Seleciona as melhores sentenças
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:3]]
        
        summary = '. '.join(top_sentences)
        if not summary.endswith('.'):
            summary += '.'
            
        return summary
    
    def _extract_portuguese_topics(self, text: str) -> List[str]:
        """
        Extrai tópicos OTIMIZADOS para português brasileiro.
        Identifica termos e frases relevantes comuns em reuniões corporativas brasileiras.
        """
        logger.info("🏷️ Extraindo tópicos em português brasileiro")
        
        # Palavras-chave específicas para reuniões em português brasileiro
        topic_indicators = [
            'agenda', 'ponto', 'item', 'assunto', 'tema', 'tópico',
            'projeto', 'proposta', 'plano', 'estratégia', 'meta', 'objetivo',
            'problema', 'solução', 'decisão', 'definição', 'acordo',
            'responsabilidade', 'ação', 'tarefa', 'próximo passo',
            'prazo', 'cronograma', 'deadline', 'entrega',
            'orçamento', 'custo', 'investimento', 'recurso',
            'cliente', 'parceiro', 'fornecedor', 'equipe', 'time'
        ]
        
        # Separa o texto em sentenças
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
        topics = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Verifica se a sentença contém indicadores de tópico
            has_topic_indicator = any(indicator in sentence_lower for indicator in topic_indicators)
            
            # Critérios para identificar um tópico válido em PT-BR
            word_count = len(sentence.split())
            
            if (has_topic_indicator and 
                word_count >= 4 and 
                word_count <= 20 and 
                len(sentence) >= 20 and
                len(sentence) <= 150):
                
                # Limpa e formata o tópico
                topic = sentence.strip()
                
                # Remove caracteres indesejados do início
                topic = topic.lstrip(',-;:')
                
                # Capitaliza a primeira letra
                if topic and topic[0].islower():
                    topic = topic[0].upper() + topic[1:]
                
                # Adiciona ponto final se necessário
                if topic and not topic.endswith(('.', '!', '?')):
                    topic += '.'
                
                topics.append(topic)
                
                # Limita a 5 tópicos para não sobrecarregar
                if len(topics) >= 5:
                    break
        
        # Se não encontrou tópicos específicos, usa estratégia de backup
        if not topics:
            logger.info("🔄 Usando estratégia de backup para extração de tópicos")
            backup_topics = self._extract_backup_topics(text)
            topics.extend(backup_topics)
        
        # Remove duplicatas mantendo a ordem
        unique_topics = []
        seen = set()
        for topic in topics:
            topic_key = topic.lower().replace('.', '').strip()
            if topic_key not in seen and len(topic_key) > 10:
                unique_topics.append(topic)
                seen.add(topic_key)
        
        result_topics = unique_topics[:3]  # Máximo de 3 tópicos principais
        logger.info(f"✅ Tópicos extraídos: {len(result_topics)} tópicos")
        
        return result_topics
    
    def _extract_backup_topics(self, text: str) -> List[str]:
        """
        Método de backup para extração de tópicos quando outros métodos falham.
        """
        logger.info("🔄 Usando método de backup para extração de tópicos")
        
        # Palavras-chave básicas para identificar tópicos
        basic_keywords = ['projeto', 'problema', 'solução', 'decisão', 'ação', 'prazo']
        
        words = text.lower().split()
        topics = []
        
        for keyword in basic_keywords:
            if keyword in words:
                # Encontra contexto ao redor da palavra-chave
                for i, word in enumerate(words):
                    if word == keyword:
                        start = max(0, i-2)
                        end = min(len(words), i+3)
                        context = ' '.join(words[start:end])
                        if len(context) > 10:
                            topics.append(context.capitalize())
                        break
        
        return topics[:5] if topics else ["Reunião geral"]

    async def _get_existing_analysis(self, meeting_id: int) -> Optional[MeetingAnalysisResult]:
        """Busca análise existente da reunião no banco de dados."""
        try:
            async with get_db() as db:
                analysis = await db.meetinganalysis.find_first(
                    where={"meeting_id": meeting_id},
                    order={"generated_at": "desc"}
                )
                
                if not analysis:
                    return None
                
                # Reconstrói o resultado da análise
                participants = []
                if analysis.participants:
                    participants_data = json.loads(analysis.participants)
                    from app.schemas.transcription import ParticipantInfo
                    participants = [ParticipantInfo(**p) for p in participants_data]
                
                main_topics = []
                if analysis.main_topics:
                    topics_data = json.loads(analysis.main_topics)
                    from app.schemas.transcription import TopicInfo
                    main_topics = [TopicInfo(**t) for t in topics_data]
                
                action_items = []
                if analysis.action_items:
                    actions_data = json.loads(analysis.action_items)
                    from app.schemas.transcription import ActionItem
                    action_items = [ActionItem(**a) for a in actions_data]
                
                key_decisions = []
                if analysis.key_decisions:
                    decisions_data = json.loads(analysis.key_decisions)
                    from app.schemas.transcription import KeyDecision
                    key_decisions = [KeyDecision(**d) for d in decisions_data]
                
                sentiment_analysis = None
                if analysis.sentiment_analysis:
                    sentiment_data = json.loads(analysis.sentiment_analysis)
                    from app.schemas.transcription import SentimentAnalysis
                    sentiment_analysis = SentimentAnalysis(**sentiment_data)
                
                return MeetingAnalysisResult(
                    participants=participants,
                    main_topics=main_topics,
                    action_items=action_items,
                    key_decisions=key_decisions,
                    summary=analysis.summary or "",
                    sentiment_analysis=sentiment_analysis,
                    confidence_score=analysis.confidence_score or 0.8
                )
                
        except Exception as e:
            logger.error(f"Erro ao buscar análise existente: {e}")
            return None

    async def _get_existing_summary(self, meeting_id: int) -> Optional[str]:
        """Busca resumo existente da reunião no banco de dados."""
        try:
            async with get_db() as db:
                summary = await db.summary.find_first(
                    where={"meeting_id": meeting_id},
                    order={"generated_at": "desc"}
                )
                return summary.content if summary else None
        except Exception as e:
            logger.error(f"Erro ao buscar resumo existente: {e}")
            return None

    async def _get_existing_topics(self, meeting_id: int) -> List[str]:
        """Busca tópicos existentes da reunião no banco de dados."""
        try:
            async with get_db() as db:
                summary = await db.summary.find_first(
                    where={"meeting_id": meeting_id},
                    order={"generated_at": "desc"}
                )
                if summary and summary.topics:
                    return json.loads(summary.topics)
                return []
        except Exception as e:
            logger.error(f"Erro ao buscar tópicos existentes: {e}")
            return []

    def _transcribe_long_audio_optimized(self, audio_data: np.ndarray, sample_rate: int, config: dict, task_id: Optional[str] = None) -> str:
        """
        ✅ OTIMIZADO: Transcreve áudios longos com configurações adaptativas
        """
        max_duration = config["chunk_duration"]
        logger.info(f"🔄 Transcrevendo áudio longo OTIMIZADO em segmentos de {max_duration}s")
        
        chunk_size = int(max_duration * sample_rate)
        total_samples = len(audio_data)
        transcriptions = []
        
        # Calcula número total de chunks
        total_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        # Atualiza progresso com informações de chunks
        if task_id:
            progress_service.update_transcription_chunks(task_id, total_chunks, 0)
        
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk = audio_data[i:end_idx]
            
            # Evita chunks muito pequenos (menos de 1 segundo)
            chunk_duration = len(chunk) / sample_rate
            if chunk_duration < 1.0:
                logger.info(f"Chunk muito pequeno ({chunk_duration:.2f}s), pulando")
                continue
            
            chunk_number = i//chunk_size + 1
            logger.info(f"⚡ Transcrevendo chunk OTIMIZADO {chunk_number}: {chunk_duration:.2f}s")
            
            # Atualiza progresso do chunk atual
            if task_id:
                progress_service.update_transcription_chunks(task_id, total_chunks, chunk_number - 1)
            
            try:
                chunk_transcription = self._generate_transcription_optimized(chunk, sample_rate, config)
                if chunk_transcription.strip():
                    transcriptions.append(chunk_transcription.strip())
                    logger.info(f"✅ Chunk transcrito: {len(chunk_transcription)} caracteres")
                    
                    # Atualiza progresso após completar o chunk
                    if task_id:
                        progress_service.update_transcription_chunks(task_id, total_chunks, chunk_number)
                else:
                    logger.info("⚠️ Chunk sem transcrição detectada")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao transcrever chunk {chunk_number}: {e}")
                continue
        
        # Junta todas as transcrições
        if transcriptions:
            full_transcription = " ".join(transcriptions)
            logger.info(f"✅ Transcrição OTIMIZADA completa: {len(transcriptions)} chunks, {len(full_transcription)} caracteres")
            return full_transcription
        else:
            logger.warning("⚠️ Nenhuma transcrição válida gerada")
            return "[Áudio sem fala detectada ou muito baixo]"

    def _transcribe_long_audio(self, audio_data: np.ndarray, sample_rate: int, max_duration: float = 30.0, task_id: Optional[str] = None) -> str:
        """
        Transcreve áudios longos dividindo em segmentos menores para evitar problemas de limite de tokens.
        """
        logger.info(f"🔄 Transcrevendo áudio longo em segmentos de {max_duration}s")
        
        chunk_size = int(max_duration * sample_rate)
        total_samples = len(audio_data)
        transcriptions = []
        
        # Calcula número total de chunks
        total_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        # Atualiza progresso com informações de chunks
        if task_id:
            progress_service.update_transcription_chunks(task_id, total_chunks, 0)
        
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk = audio_data[i:end_idx]
            
            # Evita chunks muito pequenos (menos de 1 segundo)
            chunk_duration = len(chunk) / sample_rate
            if chunk_duration < 1.0:
                logger.info(f"Chunk muito pequeno ({chunk_duration:.2f}s), pulando")
                continue
            
            chunk_number = i//chunk_size + 1
            logger.info(f"Transcrevendo chunk {chunk_number}: {chunk_duration:.2f}s")
            
            # Atualiza progresso do chunk atual
            if task_id:
                progress_service.update_transcription_chunks(task_id, total_chunks, chunk_number - 1)
            
            try:
                chunk_transcription = self._generate_transcription(chunk, sample_rate)
                if chunk_transcription.strip():
                    transcriptions.append(chunk_transcription.strip())
                    logger.info(f"✅ Chunk transcrito: {len(chunk_transcription)} caracteres")
                    
                    # Atualiza progresso após completar o chunk
                    if task_id:
                        progress_service.update_transcription_chunks(task_id, total_chunks, chunk_number)
                else:
                    logger.info("⚠️ Chunk sem transcrição detectada")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao transcrever chunk {chunk_number}: {e}")
                continue
        
        # Junta todas as transcrições
        if transcriptions:
            full_transcription = " ".join(transcriptions)
            logger.info(f"✅ Transcrição completa: {len(transcriptions)} chunks, {len(full_transcription)} caracteres")
            return full_transcription
        else:
            logger.warning("⚠️ Nenhuma transcrição válida gerada")
            return "[Áudio sem fala detectada ou muito baixo]"

    def _generate_transcription_optimized(self, audio_data: np.ndarray, sample_rate: int, config: dict) -> str:
        """
        ✅ OTIMIZADO: Gera transcrição com configurações adaptativas para máxima velocidade
        """
        try:
            logger.info("⚡ Iniciando transcrição OTIMIZADA com Whisper")
            
            # Força o carregamento do modelo
            model = self.transcription_model
            processor = self.processor
            
            logger.info(f"📊 Whisper carregado: {type(model)}")
            logger.info(f"🎛️ Processor carregado: {type(processor)}")
            
            # Verifica se o modelo foi carregado
            if self._transcription_model is None or self._processor is None:
                raise RuntimeError("Whisper não foi carregado corretamente")
            
            logger.info(f"🔊 Processando áudio: shape={audio_data.shape}, sample_rate={sample_rate}")
            
            # Processa o áudio
            inputs = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            logger.info(f"✅ Inputs processados: {list(inputs.keys())}")
            
            # Gera a transcrição
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🖥️ Usando device: {device}")
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # ✅ OTIMIZAÇÃO: Usar configurações adaptativas do config
                generation_kwargs = {
                    "max_new_tokens": config["max_new_tokens"],
                    "num_beams": config["num_beams"],
                    "do_sample": False,
                    "early_stopping": config["early_stopping"],
                    "language": "portuguese",
                    "task": "transcribe"
                }
                
                logger.info(f"🎯 Configuração OTIMIZADA: {generation_kwargs}")
                
                try:
                    outputs = self.transcription_model.generate(**inputs, **generation_kwargs)
                    logger.info("✅ Transcrição gerada com configurações OTIMIZADAS")
                except (TypeError, ValueError) as e:
                    # Fallback para configuração ainda mais rápida
                    logger.warning(f"⚠️ Configurações não suportadas: {e}")
                    logger.info("🔄 Usando configuração ULTRA rápida")
                    
                    fallback_kwargs = {
                        "max_new_tokens": 150,
                        "num_beams": 1,
                        "do_sample": False,
                        "early_stopping": True
                    }
                    try:
                        outputs = self.transcription_model.generate(**inputs, **fallback_kwargs)
                        logger.info("✅ Transcrição gerada com configuração ULTRA rápida")
                    except Exception as fallback_error:
                        logger.warning(f"⚠️ Fallback também falhou: {fallback_error}")
                        # Último recurso - configuração mínima
                        outputs = self.transcription_model.generate(**inputs, max_new_tokens=100, num_beams=1)
                
                transcription_text = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Pós-processamento para português brasileiro
            transcription_text = self._postprocess_portuguese_text(transcription_text)
            
            logger.info(f"✅ Transcrição OTIMIZADA gerada: {transcription_text[:100]}...")
            return transcription_text
            
        except Exception as e:
            logger.error(f"❌ Erro na transcrição OTIMIZADA: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Erro na geração de transcrição OTIMIZADA: {str(e)}")

    def _generate_transcription(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Gera transcrição usando APENAS Whisper Large-v3 para português brasileiro.
        """
        try:
            logger.info("🎙️ Iniciando transcrição com Whisper Large-v3")
            
            # Força o carregamento do modelo
            model = self.transcription_model
            processor = self.processor
            
            logger.info(f"📊 Whisper Large-v3 carregado: {type(model)}")
            logger.info(f"🎛️ Processor carregado: {type(processor)}")
            
            # Verifica se o modelo foi carregado
            if self._transcription_model is None or self._processor is None:
                raise RuntimeError("Whisper Large-v3 não foi carregado corretamente")
            
            logger.info(f"🔊 Processando áudio: shape={audio_data.shape}, sample_rate={sample_rate}")
            
            # Processa o áudio
            inputs = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            logger.info(f"✅ Inputs processados: {list(inputs.keys())}")
            
            # Gera a transcrição
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🖥️ Usando device: {device}")
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Configurações OTIMIZADAS para Whisper Large-v3 em português
                # Otimizações aplicadas para 2-3x mais velocidade:
                # - Reduzido num_beams de 5 para 2 (2.5x mais rápido)
                # - Reduzido max_new_tokens para 250 (mais eficiente)
                # - Adicionado early_stopping para parada antecipada
                generation_kwargs = {
                    "max_new_tokens": 250,      # ✅ Otimizado: reduzido de 440 para 250
                    "num_beams": 2,             # ✅ Otimizado: reduzido de 5 para 2 (2.5x mais rápido)
                    "do_sample": False,
                    "early_stopping": True,     # ✅ Otimizado: parada antecipada para mais velocidade
                    "language": "portuguese",   # Força português brasileiro
                    "task": "transcribe"        # Força transcrição (não tradução)
                }
                
                try:
                    outputs = self.transcription_model.generate(**inputs, **generation_kwargs)
                    logger.info("✅ Transcrição gerada com configurações PT-BR")
                except (TypeError, ValueError) as e:
                    # Fallback para Whisper Large-v3 sem configurações específicas
                    logger.warning(f"⚠️ Configurações PT-BR não suportadas: {e}")
                    logger.info("🔄 Usando Whisper Large-v3 com configuração padrão")
                    
                    # Fallback OTIMIZADO - ainda mais rápido
                    fallback_kwargs = {
                        "max_new_tokens": 200,  # ✅ Otimizado: ainda mais conservador
                        "num_beams": 2,         # ✅ Otimizado: reduzido para 2 beams
                        "do_sample": False,
                        "early_stopping": True  # ✅ Otimizado: parada antecipada
                    }
                    try:
                        outputs = self.transcription_model.generate(**inputs, **fallback_kwargs)
                        logger.info("✅ Transcrição gerada com configuração fallback")
                    except Exception as fallback_error:
                        logger.warning(f"⚠️ Fallback também falhou: {fallback_error}")
                        # Último recurso - configuração ULTRA rápida
                        outputs = self.transcription_model.generate(**inputs, max_new_tokens=150, num_beams=1)
                
                transcription_text = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Pós-processamento para português brasileiro
            transcription_text = self._postprocess_portuguese_text(transcription_text)
            
            logger.info(f"✅ Transcrição Whisper Large-v3 gerada: {transcription_text[:100]}...")
            return transcription_text
            
        except Exception as e:
            logger.error(f"❌ Erro na transcrição Whisper Large-v3: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Erro na geração de transcrição com Whisper Large-v3: {str(e)}")
    
    def _postprocess_portuguese_text(self, text: str) -> str:
        """
        Pós-processa o texto transcrito para melhorar a qualidade em português brasileiro.
        """
        if not text:
            return text
            
        # Remove espaços extras
        text = ' '.join(text.split())
        
        # Correções comuns para transcrições em português
        corrections = {
            ' né ': ' né? ',
            ' tá ': ' está ',
            ' pra ': ' para ',
            ' pro ': ' para o ',
            ' vc ': ' você ',
            ' vcs ': ' vocês ',
            'ok ': 'OK ',
            'email': 'e-mail',
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # Garante que termina com pontuação
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Capitaliza primeira letra
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text 

# Instância global do serviço de transcrição
transcription_service = TranscriptionService() 