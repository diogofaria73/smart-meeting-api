import asyncio
import json
import re
import time
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from app.schemas.transcription import (
    MeetingAnalysisResult, ParticipantInfo, TopicInfo, ActionItem, 
    KeyDecision, SentimentAnalysis, Priority
)
from app.services.progress_service import progress_service
from app.db.client import get_db

logger = logging.getLogger(__name__)


class EnhancedSummaryService:
    """
    🚀 Serviço de sumarização otimizado com pipeline assíncrono.
    
    Funcionalidades:
    - ⚡ Processamento em background com WebSocket
    - 🧠 Pipeline configurável e modular
    - ⚡ Cache inteligente de resultados
    - 📊 Métricas de qualidade e performance
    - 🔄 Chunking inteligente para textos longos
    - 🎯 Análise semântica avançada
    """
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1 hora
        
        # Pipeline de análise configurável
        self.analysis_pipeline = [
            'preprocess_text',
            'chunk_text',
            'extract_participants',
            'extract_topics',
            'extract_actions',
            'extract_decisions',
            'analyze_sentiment',
            'generate_summary',
            'calculate_confidence'
        ]
        
        # Configurações otimizadas
        self.config = {
            'max_chunk_size': 1000,  # Caracteres por chunk
            'chunk_overlap': 100,    # Sobreposição entre chunks
            'min_confidence': 0.6,   # Confiança mínima
            'max_participants': 10,  # Máximo de participantes
            'max_topics': 5,         # Máximo de tópicos
            'max_actions': 15,       # Máximo de ações
            'parallel_processing': True,  # Processamento paralelo
            'cache_enabled': True,   # Cache habilitado
        }
        
        # Padrões otimizados com pesos
        self.participant_patterns = [
            (r'(?:meu\s+nome\s+é|eu\s+sou\s+(?:o|a)?\s*)([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)', 3.0),
            (r'([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)\s+(?:disse|falou|comentou|mencionou)', 2.0),
            (r'(?:como\s+(?:disse|falou)\s+(?:o|a)?\s*)([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)', 1.5),
            (r'([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)\s+(?:explicou|apresentou|propôs)', 2.5),
        ]
        
        self.action_patterns = [
            (r'([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)\s+(?:vai|deve|precisa|fica\s+responsável)\s+([^.!?]+)', 3.0),
            (r'(?:precisa|deve|vai|tem\s+que)\s+([^.!?]+)', 2.0),
            (r'(?:ação|tarefa|atividade):\s*([^.!?]+)', 2.5),
            (r'(?:entregar|enviar|preparar|revisar|fazer|executar)\s+([^.!?]+)', 2.0),
        ]
        
        self.decision_patterns = [
            (r'(?:decidiu-se|ficou\s+decidido|foi\s+aprovado|concordamos)\s+([^.!?]+)', 3.0),
            (r'(?:decisão|resolução|acordo):\s*([^.!?]+)', 2.5),
            (r'(?:vamos\s+adotar|optamos\s+por|escolhemos)\s+([^.!?]+)', 2.0),
        ]
        
        # Palavras-chave com pesos
        self.topic_keywords = {
            'projeto': 3.0, 'orçamento': 2.8, 'estratégia': 2.7, 'plano': 2.5,
            'meta': 2.3, 'objetivo': 2.2, 'problema': 2.8, 'solução': 2.7,
            'cliente': 2.5, 'parceiro': 2.0, 'equipe': 2.0, 'cronograma': 2.6,
            'prazo': 2.4, 'entrega': 2.3, 'produto': 2.1, 'serviço': 2.0
        }

    async def analyze_meeting_async(
        self, 
        meeting_id: int,
        transcription_text: str,
        task_id: Optional[str] = None,
        custom_config: Optional[Dict] = None
    ) -> MeetingAnalysisResult:
        """
        🚀 Análise assíncrona otimizada com progresso em tempo real.
        """
        start_time = time.time()
        
        # Mescla configuração customizada
        config = {**self.config, **(custom_config or {})}
        
        # Verifica cache primeiro
        cache_key = self._generate_cache_key(transcription_text, config)
        if config['cache_enabled'] and cache_key in self.analysis_cache:
            cached_result, timestamp = self.analysis_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info("✅ Retornando resultado do cache")
                if task_id:
                    progress_service.update_progress(task_id, 'analysis_cache', 'Resultado encontrado no cache', 100)
                return cached_result

        logger.info(f"🔍 Iniciando análise assíncrona otimizada para reunião {meeting_id}")
        
        try:
            # Atualiza progresso inicial
            if task_id:
                progress_service.update_progress(task_id, 'analysis_start', 'Iniciando análise inteligente...', 5)
            
            # Executa pipeline de análise
            result = await self._execute_analysis_pipeline(
                transcription_text, config, task_id
            )
            
            # Adiciona métricas de performance
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Salva no cache
            if config['cache_enabled']:
                self.analysis_cache[cache_key] = (result, time.time())
                self._cleanup_cache()
            
            if task_id:
                progress_service.update_progress(task_id, 'analysis_complete', 'Análise concluída com sucesso!', 100)
            
            logger.info(f"🎯 Análise otimizada concluída em {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na análise assíncrona: {str(e)}")
            if task_id:
                progress_service.mark_failed(task_id, f"Erro na análise: {str(e)}")
            raise

    async def _execute_analysis_pipeline(
        self, 
        text: str, 
        config: Dict, 
        task_id: Optional[str] = None
    ) -> MeetingAnalysisResult:
        """
        🔄 Executa pipeline de análise com etapas configuráveis.
        """
        pipeline_data = {'original_text': text, 'config': config}
        total_steps = len(self.analysis_pipeline)
        
        for i, step_name in enumerate(self.analysis_pipeline):
            step_progress = int((i / total_steps) * 90) + 10  # 10-100%
            
            if task_id:
                progress_service.update_progress(
                    task_id, 
                    f'analysis_{step_name}', 
                    f'Executando: {step_name.replace("_", " ").title()}...', 
                    step_progress
                )
            
            # Executa etapa do pipeline
            step_method = getattr(self, f'_pipeline_{step_name}')
            pipeline_data = await step_method(pipeline_data)
            
            logger.info(f"✅ Pipeline step '{step_name}' concluída")
        
        # Constrói resultado final
        return MeetingAnalysisResult(
            participants=pipeline_data.get('participants', []),
            main_topics=pipeline_data.get('topics', []),
            action_items=pipeline_data.get('actions', []),
            key_decisions=pipeline_data.get('decisions', []),
            summary=pipeline_data.get('summary', ''),
            sentiment_analysis=pipeline_data.get('sentiment', None),
            confidence_score=pipeline_data.get('confidence', 0.0),
            processing_time=pipeline_data.get('processing_time', 0.0)
        )

    async def _pipeline_preprocess_text(self, data: Dict) -> Dict:
        """📝 Pré-processamento otimizado do texto."""
        text = data['original_text']
        
        # Remove ruídos comuns em transcrições
        text = re.sub(r'\[.*?\]', '', text)  # Remove timestamps
        text = re.sub(r'\b(?:uh|um|eh|ah)\b', '', text, flags=re.IGNORECASE)  # Remove hesitações
        text = re.sub(r'\s+', ' ', text)  # Normaliza espaços
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Corrige pontuação
        
        # Remove linhas muito curtas ou repetitivas
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not self._is_repetitive_line(line):
                cleaned_lines.append(line)
        
        data['processed_text'] = ' '.join(cleaned_lines)
        return data

    async def _pipeline_chunk_text(self, data: Dict) -> Dict:
        """🧩 Divisão inteligente do texto em chunks."""
        text = data['processed_text']
        config = data['config']
        
        chunk_size = config['max_chunk_size']
        overlap = config['chunk_overlap']
        
        chunks = []
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Salva chunk atual
                chunks.append('. '.join(current_chunk) + '.')
                
                # Inicia novo chunk com sobreposição
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Adiciona último chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        data['chunks'] = chunks
        logger.info(f"📦 Texto dividido em {len(chunks)} chunks")
        return data

    async def _pipeline_extract_participants(self, data: Dict) -> Dict:
        """👥 Extração otimizada de participantes com processamento paralelo."""
        chunks = data['chunks']
        config = data['config']
        
        if config['parallel_processing']:
            # Processamento paralelo dos chunks
            tasks = [
                asyncio.create_task(self._extract_participants_from_chunk(chunk))
                for chunk in chunks
            ]
            chunk_results = await asyncio.gather(*tasks)
        else:
            # Processamento sequencial
            chunk_results = []
            for chunk in chunks:
                result = await self._extract_participants_from_chunk(chunk)
                chunk_results.append(result)
        
        # Combina resultados dos chunks
        participants = self._merge_participant_results(chunk_results, config)
        
        data['participants'] = participants
        logger.info(f"👥 {len(participants)} participantes extraídos")
        return data

    async def _extract_participants_from_chunk(self, chunk: str) -> List[Tuple[str, float, str]]:
        """Extrai participantes de um chunk específico."""
        participants = []
        
        for pattern, weight in self.participant_patterns:
            matches = list(re.finditer(pattern, chunk, re.IGNORECASE))
            for match in matches:
                name = match.group(1).strip()
                name = self._normalize_name(name)
                
                if name and len(name) > 2:
                    # Extrai contexto
                    start = max(0, match.start() - 50)
                    end = min(len(chunk), match.end() + 50)
                    context = chunk[start:end].strip()
                    
                    participants.append((name, weight, context))
        
        return participants

    def _merge_participant_results(self, chunk_results: List, config: Dict) -> List[ParticipantInfo]:
        """Combina resultados de participantes de múltiplos chunks."""
        participant_data = defaultdict(lambda: {'mentions': 0, 'weight_sum': 0.0, 'contexts': []})
        
        for chunk_result in chunk_results:
            for name, weight, context in chunk_result:
                participant_data[name]['mentions'] += 1
                participant_data[name]['weight_sum'] += weight
                participant_data[name]['contexts'].append(context)
        
        participants = []
        for name, data in participant_data.items():
            if data['mentions'] >= 1:
                avg_weight = data['weight_sum'] / data['mentions']
                confidence = min(0.95, 0.4 + (avg_weight * 0.2) + (data['mentions'] * 0.1))
                
                if confidence >= config['min_confidence']:
                    role = self._extract_participant_role(data['contexts'])
                    
                    participants.append(ParticipantInfo(
                        name=name,
                        mentions=data['mentions'],
                        role=role,
                        confidence=confidence
                    ))
        
        # Ordena por confiança e mencões
        participants.sort(key=lambda p: (p.confidence, p.mentions), reverse=True)
        return participants[:config['max_participants']]

    # Continua com outras etapas do pipeline...
    async def _pipeline_extract_topics(self, data: Dict) -> Dict:
        """📋 Extração avançada de tópicos com análise semântica."""
        chunks = data['chunks']
        config = data['config']
        
        # Processa cada chunk para extrair tópicos
        all_topics = []
        for chunk in chunks:
            chunk_topics = await self._extract_topics_from_chunk(chunk)
            all_topics.extend(chunk_topics)
        
        # Agrupa e ranqueia tópicos similares
        merged_topics = self._merge_and_rank_topics(all_topics, config)
        
        data['topics'] = merged_topics
        logger.info(f"📋 {len(merged_topics)} tópicos principais extraídos")
        return data

    async def _extract_topics_from_chunk(self, chunk: str) -> List[Dict]:
        """Extrai tópicos de um chunk específico."""
        topics = []
        
        # Analisa cada sentença do chunk
        sentences = [s.strip() for s in chunk.split('.') if len(s.strip()) > 15]
        
        for sentence in sentences:
            # Calcula score baseado em palavras-chave
            topic_score = 0
            sentence_lower = sentence.lower()
            matched_keywords = []
            
            for keyword, weight in self.topic_keywords.items():
                if keyword in sentence_lower:
                    topic_score += weight
                    matched_keywords.append(keyword)
            
            # Se tem score suficiente, considera como tópico
            if topic_score >= 2.0 and len(sentence) <= 150:
                topics.append({
                    'title': sentence.strip(),
                    'score': topic_score,
                    'keywords': matched_keywords,
                    'length': len(sentence)
                })
        
        return topics

    def _merge_and_rank_topics(self, all_topics: List[Dict], config: Dict) -> List[TopicInfo]:
        """Agrupa tópicos similares e ranqueia por importância."""
        # Agrupa tópicos similares (implementação simplificada)
        unique_topics = []
        
        for topic in all_topics:
            # Verifica se já existe tópico similar
            is_duplicate = False
            for existing in unique_topics:
                if self._calculate_topic_similarity(topic['title'], existing['title']) > 0.7:
                    # Mescla com tópico existente
                    existing['score'] += topic['score']
                    existing['keywords'].extend(topic['keywords'])
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_topics.append(topic)
        
        # Converte para TopicInfo e ordena
        topic_infos = []
        for topic in unique_topics:
            # Remove keywords duplicadas
            unique_keywords = list(set(topic['keywords']))
            
            topic_infos.append(TopicInfo(
                title=topic['title'],
                summary=topic['title'],  # Simplificado
                keywords=unique_keywords,
                importance=min(1.0, topic['score'] / 10.0),
                duration_mentioned="~1min"  # Estimativa
            ))
        
        # Ordena por importância
        topic_infos.sort(key=lambda t: t.importance, reverse=True)
        return topic_infos[:config['max_topics']]

    def _calculate_topic_similarity(self, title1: str, title2: str) -> float:
        """Calcula similaridade entre dois títulos de tópicos."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    # Pipeline steps continuam...
    async def _pipeline_extract_actions(self, data: Dict) -> Dict:
        """✅ Extração otimizada de itens de ação."""
        chunks = data['chunks']
        participants = data.get('participants', [])
        config = data['config']
        
        all_actions = []
        participant_names = [p.name for p in participants]
        
        for chunk in chunks:
            chunk_actions = await self._extract_actions_from_chunk(chunk, participant_names)
            all_actions.extend(chunk_actions)
        
        # Remove duplicatas e ranqueia
        unique_actions = self._deduplicate_and_rank_actions(all_actions, config)
        
        data['actions'] = unique_actions
        logger.info(f"✅ {len(unique_actions)} itens de ação extraídos")
        return data

    async def _extract_actions_from_chunk(self, chunk: str, participant_names: List[str]) -> List[ActionItem]:
        """Extrai ações de um chunk específico."""
        actions = []
        
        for pattern, weight in self.action_patterns:
            matches = list(re.finditer(pattern, chunk, re.IGNORECASE))
            for match in matches:
                if len(match.groups()) >= 2:
                    assignee = match.group(1).strip()
                    task = match.group(2).strip()
                else:
                    assignee = None
                    task = match.group(1).strip()
                
                # Limpa o texto da tarefa
                task = self._clean_task_text(task)
                if not task:
                    continue
                
                # Extrai prazo se possível
                due_date = self._extract_due_date_from_context(chunk, match.start(), match.end())
                
                # Determina prioridade
                priority = self._determine_priority_enhanced(task, chunk)
                
                # Calcula confiança
                confidence = self._calculate_action_confidence_enhanced(task, assignee, due_date, weight)
                
                if confidence >= 0.5:
                    actions.append(ActionItem(
                        task=task,
                        assignee=assignee,
                        due_date=due_date,
                        priority=priority,
                        confidence=confidence
                    ))
        
        return actions

    def _deduplicate_and_rank_actions(self, actions: List[ActionItem], config: Dict) -> List[ActionItem]:
        """Remove duplicatas e ranqueia ações por importância."""
        unique_actions = []
        
        for action in actions:
            # Verifica se já existe ação similar
            is_duplicate = False
            for existing in unique_actions:
                if self._calculate_action_similarity(action.task, existing.task) > 0.8:
                    # Mantém a ação com maior confiança
                    if action.confidence > existing.confidence:
                        unique_actions.remove(existing)
                        unique_actions.append(action)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_actions.append(action)
        
        # Ordena por prioridade e confiança
        unique_actions.sort(key=lambda a: (a.priority.value, a.confidence), reverse=True)
        return unique_actions[:config['max_actions']]

    def _calculate_action_similarity(self, task1: str, task2: str) -> float:
        """Calcula similaridade entre duas tarefas."""
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    # Continua com outros métodos do pipeline...
    async def _pipeline_extract_decisions(self, data: Dict) -> Dict:
        """⚖️ Extração de decisões importantes."""
        data['decisions'] = []  # Implementação simplificada
        return data

    async def _pipeline_analyze_sentiment(self, data: Dict) -> Dict:
        """😊 Análise de sentimento."""
        data['sentiment'] = None  # Implementação simplificada
        return data

    async def _pipeline_generate_summary(self, data: Dict) -> Dict:
        """📄 Geração de resumo estruturado."""
        participants = data.get('participants', [])
        topics = data.get('topics', [])
        actions = data.get('actions', [])
        
        summary_parts = []
        
        # Seção de participantes
        if participants:
            participant_names = [p.name for p in participants[:3]]
            summary_parts.append(f"Participantes principais: {', '.join(participant_names)}")
        
        # Seção de tópicos
        if topics:
            topic_titles = [t.title[:50] + '...' if len(t.title) > 50 else t.title for t in topics[:3]]
            summary_parts.append(f"Tópicos discutidos: {'; '.join(topic_titles)}")
        
        # Seção de ações
        if actions:
            action_count = len(actions)
            summary_parts.append(f"Itens de ação identificados: {action_count}")
        
        data['summary'] = '. '.join(summary_parts) + '.'
        return data

    async def _pipeline_calculate_confidence(self, data: Dict) -> Dict:
        """🎯 Cálculo da confiança geral."""
        participants = data.get('participants', [])
        topics = data.get('topics', [])
        actions = data.get('actions', [])
        
        # Calcula confiança baseada na qualidade dos resultados
        confidence_scores = []
        
        if participants:
            avg_participant_confidence = sum(p.confidence for p in participants) / len(participants)
            confidence_scores.append(avg_participant_confidence)
        
        if topics:
            avg_topic_importance = sum(t.importance for t in topics) / len(topics)
            confidence_scores.append(avg_topic_importance)
        
        if actions:
            avg_action_confidence = sum(a.confidence for a in actions) / len(actions)
            confidence_scores.append(avg_action_confidence)
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        data['confidence'] = min(0.95, overall_confidence)
        
        return data

    # Métodos auxiliares
    def _generate_cache_key(self, text: str, config: Dict) -> str:
        """Gera chave única para cache baseada no conteúdo e configuração."""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return f"{content_hash}_{config_hash}"

    def _cleanup_cache(self):
        """Remove entradas antigas do cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.analysis_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.analysis_cache[key]

    def _is_repetitive_line(self, line: str) -> bool:
        """Verifica se uma linha é repetitiva ou ruído."""
        # Implementação simplificada
        words = line.split()
        if len(words) < 3:
            return True
        
        # Verifica repetição de palavras
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        return max_count > len(words) * 0.6

    def _normalize_name(self, name: str) -> Optional[str]:
        """Normaliza nome de participante."""
        if not name or len(name) < 2:
            return None
        
        # Remove artigos e preposições
        words = name.split()
        filtered_words = []
        
        for word in words:
            word_clean = word.lower().strip('.,!?:;')
            if word_clean not in ['o', 'a', 'de', 'da', 'do', 'dos', 'das', 'e']:
                if len(word_clean) >= 2 and word_clean.isalpha():
                    filtered_words.append(word.capitalize())
        
        return ' '.join(filtered_words) if len(filtered_words) >= 1 else None

    def _extract_participant_role(self, contexts: List[str]) -> Optional[str]:
        """Extrai função/cargo do participante."""
        role_patterns = [
            r'(?:diretor|gerente|coordenador|supervisor|analista)',
            r'(?:CEO|CTO|CFO|CMO|VP|presidente)',
            r'(?:desenvolvedor|programador|designer|arquiteto)'
        ]
        
        all_context = ' '.join(contexts).lower()
        
        for pattern in role_patterns:
            match = re.search(pattern, all_context, re.IGNORECASE)
            if match:
                return match.group().title()
        
        return None

    def _clean_task_text(self, task: str) -> Optional[str]:
        """Limpa e valida texto de tarefa."""
        if not task or len(task) < 5:
            return None
        
        # Remove caracteres indesejados
        task = re.sub(r'[^\w\s\.\!\?\,\:\;\-\(\)]', ' ', task)
        task = re.sub(r'\s+', ' ', task).strip()
        
        # Verifica se é uma tarefa válida
        if len(task) > 200 or len(task.split()) < 2:
            return None
        
        return task

    def _extract_due_date_from_context(self, text: str, start: int, end: int) -> Optional[str]:
        """Extrai prazo do contexto."""
        # Implementação simplificada
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end]
        
        date_patterns = [
            r'(?:até|antes\s+de|prazo)\s+([^.!?]+)',
            r'(?:próxima\s+(?:semana|segunda|terça|quarta|quinta|sexta))',
            r'(?:\d{1,2}\s+de\s+\w+|\d{1,2}/\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip() if len(match.groups()) > 0 else match.group().strip()
        
        return None

    def _determine_priority_enhanced(self, task: str, context: str) -> Priority:
        """Determina prioridade da tarefa com análise aprimorada."""
        task_lower = task.lower()
        context_lower = context.lower()
        
        # Palavras que indicam alta prioridade
        high_priority_words = ['urgente', 'crítico', 'importante', 'imediato', 'asap']
        medium_priority_words = ['necessário', 'importante', 'deve', 'precisa']
        
        combined_text = f"{task_lower} {context_lower}"
        
        for word in high_priority_words:
            if word in combined_text:
                return Priority.HIGH
        
        for word in medium_priority_words:
            if word in combined_text:
                return Priority.MEDIUM
        
        return Priority.LOW

    def _calculate_action_confidence_enhanced(
        self, 
        task: str, 
        assignee: Optional[str], 
        due_date: Optional[str], 
        pattern_weight: float
    ) -> float:
        """Calcula confiança da ação com métricas aprimoradas."""
        confidence = 0.3  # Base
        
        # Peso do padrão usado
        confidence += pattern_weight * 0.1
        
        # Bônus por ter responsável
        if assignee:
            confidence += 0.2
        
        # Bônus por ter prazo
        if due_date:
            confidence += 0.1
        
        # Penalidade por tarefa muito vaga
        if len(task.split()) < 3:
            confidence -= 0.1
        
        # Bônus por verbos de ação
        action_verbs = ['fazer', 'criar', 'desenvolver', 'implementar', 'entregar', 'revisar']
        for verb in action_verbs:
            if verb in task.lower():
                confidence += 0.05
                break
        
        return min(0.95, max(0.1, confidence))


# Instância global do serviço
enhanced_summary_service = EnhancedSummaryService() 