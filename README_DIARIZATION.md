# 🎙️ Speaker Diarization - Identificação de Participantes

Este documento descreve a nova funcionalidade de **Speaker Diarization** (identificação de participantes) integrada ao Smart Meeting API.

## 📋 Visão Geral

A diarização de speakers permite identificar **quem está falando** e **quando** em gravações de reuniões, combinando:

- **Whisper Large-v3**: Transcrição precisa em português brasileiro
- **pyannote.audio**: Identificação state-of-the-art de speakers
- **Alinhamento temporal**: Sincronização entre texto e speakers
- **Estatísticas detalhadas**: Tempo de fala, participação e confiança

## 🚀 Instalação Rápida

### Opção 1: Script Automático (Recomendado)
```bash
cd smart-meeting-api
python scripts/install_diarization.py
```

### Opção 2: Manual
```bash
cd smart-meeting-api
poetry install
```

## ⚙️ Configuração

### 1. Token HuggingFace (Opcional, mas Recomendado)

Para melhor performance, configure um token do HuggingFace:

```bash
# 1. Acesse: https://huggingface.co/settings/tokens
# 2. Aceite os termos: https://huggingface.co/pyannote/speaker-diarization-3.1
# 3. Crie um arquivo .env.diarization:

echo "HUGGINGFACE_TOKEN=seu_token_aqui" > .env.diarization
```

### 2. Configurações Opcionais

```bash
# .env.diarization
ENABLE_SPEAKER_DIARIZATION=true
MIN_SPEAKERS=1
MAX_SPEAKERS=10
MIN_SEGMENT_DURATION=1.0
FORCE_DEVICE=cuda  # cuda, mps, cpu (deixe vazio para auto-detecção)
```

## 🎯 Como Usar

### 1. Endpoint de Teste
```bash
# Teste a funcionalidade
curl -X POST "http://localhost:8000/api/transcriptions/test-diarization" \
  -F "file=@sua_reuniao.mp3" \
  -F "enable_diarization=true"
```

### 2. Integração Automática
A diarização é **automaticamente aplicada** em transcrições normais quando:
- Áudio tem **>30 segundos**
- Hardware adequado disponível (GPU 6GB+ ou CPU 12GB+ RAM)

```bash
# Transcrição normal com diarização automática
curl -X POST "http://localhost:8000/api/transcriptions/transcribe?meeting_id=1" \
  -F "file=@reuniao_longa.mp3"
```

## 📊 Exemplo de Resposta

```json
{
  "status": "success",
  "audio_info": {
    "filename": "reuniao.mp3",
    "duration_seconds": 180.5,
    "sample_rate": 16000
  },
  "transcription": {
    "text": "Texto completo da transcrição...",
    "confidence": 0.92,
    "method": "whisper_plus_pyannote"
  },
  "speakers": {
    "count": 3,
    "segments": [
      {
        "start_time": 0.0,
        "end_time": 5.2,
        "speaker_id": "SPEAKER_00",
        "text": "Bom dia pessoal, vamos iniciar a reunião.",
        "confidence": 0.89
      },
      {
        "start_time": 5.8,
        "end_time": 12.1,
        "speaker_id": "SPEAKER_01", 
        "text": "Perfeito, tenho alguns pontos para discutir.",
        "confidence": 0.91
      }
    ]
  },
  "participants": [
    {
      "name": "Participante 1",
      "speaker_id": "SPEAKER_00",
      "speaking_time": 45.3,
      "segments_count": 8,
      "confidence": 0.85
    },
    {
      "name": "Participante 2", 
      "speaker_id": "SPEAKER_01",
      "speaking_time": 38.7,
      "segments_count": 6,
      "confidence": 0.83
    }
  ],
  "processing": {
    "total_time": 12.4,
    "details": {
      "transcription_time": 8.2,
      "diarization_time": 4.1
    }
  }
}
```

## 🔧 Requisitos Técnicos

### Mínimos
- **Python**: 3.10+
- **RAM**: 8GB
- **Áudio**: >5 segundos para resultados confiáveis

### Recomendados  
- **GPU**: NVIDIA com 6GB+ VRAM
- **RAM**: 16GB+
- **Áudio**: >30 segundos com múltiplos speakers
- **Formatos**: WAV, MP3, M4A, FLAC (qualidade alta)

### Formatos Suportados
- ✅ **WAV** (melhor qualidade)
- ✅ **MP3** (compatibilidade)
- ✅ **M4A** (Apple)
- ✅ **FLAC** (sem perda)
- ✅ **OGG** (código aberto)

## 🎯 Melhores Práticas

### Para Resultados Ótimos:
1. **Qualidade do áudio**: Use microfones dedicados
2. **Duração**: Mínimo 30-60 segundos
3. **Speakers distintos**: Vozes bem diferenciadas
4. **Pouco ruído**: Ambiente controlado
5. **Uma pessoa por vez**: Evite sobreposições

### Configurações por Cenário:

**Reunião Pequena (2-3 pessoas)**
```python
MIN_SPEAKERS=2
MAX_SPEAKERS=3
MIN_SEGMENT_DURATION=2.0
```

**Reunião Média (4-6 pessoas)**  
```python
MIN_SPEAKERS=2
MAX_SPEAKERS=6
MIN_SEGMENT_DURATION=1.5
```

**Reunião Grande (7+ pessoas)**
```python
MIN_SPEAKERS=3
MAX_SPEAKERS=10
MIN_SEGMENT_DURATION=1.0
```

## 🚨 Solução de Problemas

### Erro: "pyannote.audio não disponível"
```bash
# Reinstale as dependências
poetry install
poetry run pip install pyannote.audio
```

### Erro: "Falha ao carregar pipeline"
```bash
# Configure token HuggingFace
# 1. Visite: https://huggingface.co/pyannote/speaker-diarization-3.1
# 2. Aceite os termos de uso
# 3. Configure o token no .env.diarization
```

### Performance Lenta
```bash
# Verifique hardware
poetry run python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
"
```

### Muitos Speakers Falsos
- Reduza `MIN_SEGMENT_DURATION`
- Aumente `MIN_SPEAKERS` 
- Verifique qualidade do áudio

### Speakers Não Detectados
- Aumente `MAX_SPEAKERS`
- Verifique se há sobreposição de fala
- Use áudio com maior duração

## 📈 Monitoramento e Logs

```bash
# Logs detalhados
tail -f logs/app.log | grep -E "(🎙️|speaker|diarization)"

# Informações de performance
curl http://localhost:8000/api/transcriptions/test-diarization \
  -F "file=@test.wav" | jq '.service_info'
```

## 🔄 Fallback Automático

O sistema possui **fallback inteligente**:

1. **Diarização falta** → Transcrição simples
2. **Hardware insuficiente** → Whisper tradicional  
3. **Erro na diarização** → Faster-Whisper
4. **Pyannote indisponível** → Speaker único

## 🎯 Roadmap

### Próximas Funcionalidades:
- [ ] **Reconhecimento de voz**: Identificar speakers por nome
- [ ] **Análise de emoções**: Detectar sentimentos por speaker
- [ ] **Clustering**: Agrupar speakers similares
- [ ] **Dashboard**: Visualização de participação
- [ ] **Export**: Relatórios em PDF/Excel

## 💡 Dicas de Performance

### GPU Optimization:
```python
# Use precisão mista para economia de VRAM
FORCE_COMPUTE_TYPE=float16

# Para GPUs com muita VRAM (>8GB)
FORCE_COMPUTE_TYPE=float32
```

### CPU Optimization:
```python
# Use quantização para eficiência
FORCE_COMPUTE_TYPE=int8

# Limite threads para estabilidade
export OMP_NUM_THREADS=4
```

## 📞 Suporte

Para problemas ou dúvidas:

1. **Verifique logs**: `tail -f logs/app.log`
2. **Execute diagnóstico**: `python scripts/install_diarization.py`
3. **Teste isolado**: Use endpoint `/test-diarization`
4. **Issues**: Abra issue no repositório com logs completos

---

🎉 **Parabéns!** Agora você tem identificação automática de participantes em suas reuniões! 