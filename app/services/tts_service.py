import os
import tempfile
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydub import AudioSegment
import io

from gtts import gTTS
import pyttsx3
from fastapi import HTTPException, status

from app.schemas.tts import (
    TTSRequest, 
    TTSResponse, 
    UnifiedTTSRequest, 
    AvailableVoicesResponse,
    Character,
    DialogueLine,
    VoiceGender
)


class TTSService:
    """Serviço avançado para conversão de texto em áudio com suporte a conversas entre personagens"""
    
    def __init__(self):
        # Diretório para armazenar os arquivos de áudio temporários
        self.audio_dir = Path("temp_audio")
        self.audio_dir.mkdir(exist_ok=True)
        
        # Cache de engines pyttsx3 para personagens
        self._voice_engines: Dict[str, Any] = {}
        
        # Mapeamento de vozes disponíveis
        self._available_voices = self._get_available_voices()
    
    def _get_available_voices(self) -> List[Dict]:
        """Obtém lista de vozes disponíveis no sistema"""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            available_voices = []
            
            for i, voice in enumerate(voices):
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'age': getattr(voice, 'age', None),
                    'gender': getattr(voice, 'gender', None),
                    'languages': getattr(voice, 'languages', []),
                    'index': i
                }
                available_voices.append(voice_info)
                
            engine.stop()
            del engine
            return available_voices
        except Exception:
            return []
    
    async def get_available_voices(self) -> AvailableVoicesResponse:
        """Retorna vozes disponíveis no sistema"""
        return AvailableVoicesResponse(
            voices=self._available_voices,
            total_count=len(self._available_voices)
        )
    
    def _get_voice_for_character(self, character: Character) -> Dict:
        """Seleciona a melhor voz para um personagem"""
        if character.voice_id:
            # Procura por ID específico
            for voice in self._available_voices:
                if voice['id'] == character.voice_id or str(voice['index']) == character.voice_id:
                    return voice
        
        # Filtra por gênero se disponível
        suitable_voices = []
        for voice in self._available_voices:
            voice_gender = voice.get('gender', '').lower()
            if character.gender == VoiceGender.MALE and 'male' in voice_gender:
                suitable_voices.append(voice)
            elif character.gender == VoiceGender.FEMALE and 'female' in voice_gender:
                suitable_voices.append(voice)
        
        # Se não encontrar por gênero, usa qualquer voz disponível
        if not suitable_voices:
            suitable_voices = self._available_voices
        
        # Retorna a primeira voz adequada
        return suitable_voices[0] if suitable_voices else self._available_voices[0]
    
    def _create_character_audio(self, character: Character, text: str) -> AudioSegment:
        """Cria áudio para um personagem específico usando pyttsx3"""
        try:
            # Criar engine específico para o personagem
            engine = pyttsx3.init()
            
            # Configurar voz
            voice_info = self._get_voice_for_character(character)
            if voice_info:
                engine.setProperty('voice', voice_info['id'])
            
            # Configurar propriedades de voz
            engine.setProperty('rate', character.rate)
            engine.setProperty('volume', character.volume)
            
            # Criar arquivo temporário para o áudio
            temp_file = self.audio_dir / f"temp_{uuid.uuid4().hex[:8]}.wav"
            
            # Salvar áudio
            engine.save_to_file(text, str(temp_file))
            engine.runAndWait()
            engine.stop()
            del engine
            
            # Carregar áudio com pydub
            if temp_file.exists():
                audio = AudioSegment.from_wav(str(temp_file))
                temp_file.unlink()  # Remove arquivo temporário
                return audio
            else:
                raise Exception("Falha ao gerar áudio com pyttsx3")
                
        except Exception as e:
            # Fallback para gTTS se pyttsx3 falhar
            print(f"Fallback para gTTS para personagem {character.name}: {e}")
            return self._create_gtts_audio(text, character.language, False)
    
    def _create_gtts_audio(self, text: str, language: str, slow: bool) -> AudioSegment:
        """Cria áudio usando gTTS como fallback"""
        tts = gTTS(text=text, lang=language, slow=slow)
        
        # Usar BytesIO para não criar arquivo temporário
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Converter para AudioSegment
        return AudioSegment.from_mp3(audio_buffer)
    
    async def text_to_speech(self, request: TTSRequest) -> TTSResponse:
        """Conversão simples de texto em áudio (compatibilidade)"""
        unified_request = UnifiedTTSRequest(
            simple_text=request.text,
            language=request.language,
            slow=request.slow
        )
        return await self.unified_text_to_speech(unified_request)
    
    async def unified_text_to_speech(self, request: UnifiedTTSRequest) -> TTSResponse:
        """
        Conversão unificada que suporta tanto texto simples quanto conversas
        """
        try:
            start_time = time.time()
            
            # Verificar se é texto simples ou conversa
            if request.simple_text:
                # Modo texto simples
                audio = self._create_gtts_audio(
                    request.simple_text, 
                    request.language or "pt", 
                    request.slow or False
                )
                characters_used = []
                
            elif request.characters and request.dialogue:
                # Modo conversa entre personagens
                audio = await self._create_conversation_audio(request)
                characters_used = [char.name for char in request.characters]
                
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Forneça 'simple_text' ou 'characters' com 'dialogue'"
                )
            
            # Determinar formato de arquivo
            file_extension = request.output_format
            filename = f"tts_{uuid.uuid4().hex[:8]}.{file_extension}"
            filepath = self.audio_dir / filename
            
            # Salvar arquivo
            if file_extension == "mp3":
                audio.export(str(filepath), format="mp3")
            else:
                audio.export(str(filepath), format="wav")
            
            # Calcular duração
            duration = len(audio) / 1000.0  # pydub usa milissegundos
            processing_time = time.time() - start_time
            
            # Construir URL para download
            audio_url = f"/api/tts/download/{filename}"
            
            return TTSResponse(
                success=True,
                message=f"Áudio gerado com sucesso em {processing_time:.2f}s",
                audio_url=audio_url,
                filename=filename,
                duration=duration,
                characters_used=characters_used
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro interno ao gerar áudio: {str(e)}"
            )
    
    async def _create_conversation_audio(self, request: UnifiedTTSRequest) -> AudioSegment:
        """Cria áudio de conversa entre personagens"""
        final_audio = AudioSegment.empty()
        
        # Mapear personagens por nome
        characters_map = {char.name: char for char in request.characters}
        
        for line in request.dialogue:
            character = characters_map.get(line.character)
            if not character:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Personagem '{line.character}' não encontrado"
                )
            
            # Criar áudio para esta linha
            line_audio = self._create_character_audio(character, line.text)
            
            # Adicionar ao áudio final
            final_audio += line_audio
            
            # Adicionar pausa se especificada
            if line.pause_after > 0:
                pause_duration = int(line.pause_after * 1000)  # converter para ms
                silence = AudioSegment.silent(duration=pause_duration)
                final_audio += silence
        
        return final_audio
    
    def get_audio_file_path(self, filename: str) -> Optional[Path]:
        """
        Retorna o caminho do arquivo de áudio se existir
        """
        filepath = self.audio_dir / filename
        return filepath if filepath.exists() else None
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Remove arquivos de áudio antigos
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.audio_dir.glob("*"):
            if file_path.is_file() and current_time - file_path.stat().st_mtime > max_age_seconds:
                try:
                    file_path.unlink()
                except OSError:
                    pass 