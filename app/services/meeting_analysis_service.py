import json
import re
import time
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict

from app.schemas.transcription import (
    MeetingAnalysisResult, ParticipantInfo, TopicInfo, ActionItem, 
    KeyDecision, SentimentAnalysis, Priority
)

logger = logging.getLogger(__name__)


class MeetingAnalysisService:
    """
    Serviço de análise inteligente de reuniões.
    Extrai participantes, tópicos, tarefas e decisões do texto transcrito.
    """
    
    def __init__(self):
        # Padrões para identificação de participantes
        self.participant_patterns = [
            r'\b([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)\s+(?:disse|falou|comentou|mencionou|perguntou|respondeu)',
            r'(?:como\s+(?:disse|falou)\s+(?:o|a)?\s*)([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)',
            r'(?:segundo\s+(?:o|a)?\s*)([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)',
            r'([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)\s+(?:explicou|apresentou|propôs|sugeriu)',
            r'(?:meu\s+nome\s+é|eu\s+sou\s+(?:o|a)?\s*)([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)',
        ]
        
        # Padrões para identificação de tarefas/ações
        self.action_patterns = [
            r'(?:precisa|deve|vai|tem\s+que|fica\s+responsável\s+por|encarregado\s+de)\s+([^.!?]+)',
            r'([A-Z][a-záêçõü]+(?:\s+[A-Z][a-záêçõü]+)*)\s+(?:vai|deve|precisa|fica\s+responsável)\s+([^.!?]+)',
            r'(?:vamos|devemos|precisamos)\s+([^.!?]+)',
            r'(?:ação|tarefa|atividade):\s*([^.!?]+)',
            r'(?:entregar|enviar|preparar|revisar|fazer|executar|implementar|desenvolver)\s+([^.!?]+)',
        ]
        
        # Padrões para identificação de prazos
        self.deadline_patterns = [
            r'(?:até|antes\s+de|prazo\s+de?|deadline)\s+([^.!?]+)',
            r'(?:próxima\s+(?:semana|segunda|terça|quarta|quinta|sexta))',
            r'(?:amanhã|hoje|ontem)',
            r'(?:\d{1,2}\s+de\s+\w+|\d{1,2}/\d{1,2})',
            r'(?:final\s+do\s+mês|início\s+do\s+mês)',
        ]
        
        # Padrões para identificação de decisões
        self.decision_patterns = [
            r'(?:decidiu-se|ficou\s+decidido|foi\s+aprovado|concordamos)\s+([^.!?]+)',
            r'(?:decisão|resolução|acordo):\s*([^.!?]+)',
            r'(?:vamos\s+adotar|optamos\s+por|escolhemos)\s+([^.!?]+)',
        ]
        
        # Palavras-chave para tópicos importantes
        self.topic_keywords = [
            'projeto', 'orçamento', 'estratégia', 'plano', 'meta', 'objetivo',
            'problema', 'solução', 'proposta', 'apresentação', 'relatório',
            'cliente', 'parceiro', 'fornecedor', 'equipe', 'time', 'departamento',
            'produto', 'serviço', 'vendas', 'marketing', 'financeiro',
            'cronograma', 'prazo', 'entrega', 'milestone', 'fase'
        ]
        
        # Palavras para análise de sentimento
        self.positive_words = [
            'excelente', 'ótimo', 'bom', 'positivo', 'sucesso', 'aprovado',
            'concordo', 'perfeito', 'satisfeito', 'feliz', 'animado'
        ]
        
        self.negative_words = [
            'problema', 'dificuldade', 'ruim', 'negativo', 'fracasso', 'rejeitado',
            'discordo', 'preocupado', 'insatisfeito', 'frustrado', 'atrasado'
        ]
        
        # Nomes brasileiros comuns para filtrar falsos positivos
        self.common_names = {
            'joão', 'maria', 'josé', 'ana', 'antonio', 'francisca', 'carlos', 'paulo',
            'pedro', 'lucas', 'luiz', 'marcos', 'luis', 'gabriel', 'rafael', 'daniel',
            'marcelo', 'bruno', 'eduardo', 'felipe', 'rodrigo', 'manoel', 'ricardo',
            'adriana', 'juliana', 'patricia', 'sandra', 'monica', 'claudia', 'fernanda',
            'carla', 'cristina', 'rosana', 'luciana', 'marcia', 'andrea', 'leticia'
        }

    async def analyze_meeting(
        self, 
        transcription_text: str, 
        include_sentiment: bool = True,
        extract_participants: bool = True,
        extract_action_items: bool = True,
        min_confidence: float = 0.6
    ) -> MeetingAnalysisResult:
        """
        Realiza análise completa da reunião extraindo todas as informações relevantes.
        """
        start_time = time.time()
        logger.info("🔍 Iniciando análise inteligente da reunião")
        
        try:
            # Pré-processamento do texto
            cleaned_text = self._preprocess_text(transcription_text)
            
            # Extração de participantes
            participants = []
            if extract_participants:
                participants = self._extract_participants(cleaned_text, min_confidence)
                logger.info(f"✅ Participantes extraídos: {len(participants)}")
            
            # Extração de tópicos principais
            main_topics = self._extract_main_topics(cleaned_text, min_confidence)
            logger.info(f"✅ Tópicos extraídos: {len(main_topics)}")
            
            # Extração de itens de ação
            action_items = []
            if extract_action_items:
                action_items = self._extract_action_items(cleaned_text, participants, min_confidence)
                logger.info(f"✅ Itens de ação extraídos: {len(action_items)}")
            
            # Extração de decisões importantes
            key_decisions = self._extract_key_decisions(cleaned_text, min_confidence)
            logger.info(f"✅ Decisões extraídas: {len(key_decisions)}")
            
            # Geração de resumo estruturado
            summary = self._generate_structured_summary(
                cleaned_text, participants, main_topics, action_items, key_decisions
            )
            
            # Análise de sentimento
            sentiment_analysis = None
            if include_sentiment:
                sentiment_analysis = self._analyze_sentiment(cleaned_text, main_topics)
                logger.info("✅ Análise de sentimento concluída")
            
            # Cálculo da confiança geral
            confidence_score = self._calculate_overall_confidence(
                participants, main_topics, action_items, key_decisions
            )
            
            processing_time = time.time() - start_time
            
            result = MeetingAnalysisResult(
                participants=participants,
                main_topics=main_topics,
                action_items=action_items,
                key_decisions=key_decisions,
                summary=summary,
                sentiment_analysis=sentiment_analysis,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
            logger.info(f"🎯 Análise concluída em {processing_time:.2f}s com confiança {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na análise da reunião: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Pré-processa o texto removendo ruídos e normalizando."""
        # Remove caracteres especiais desnecessários
        text = re.sub(r'[^\w\s\.\!\?\,\:\;\-\(\)]', ' ', text)
        
        # Normaliza espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Remove linhas muito curtas (provavelmente ruído)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return ' '.join(cleaned_lines)
    
    def _extract_participants(self, text: str, min_confidence: float) -> List[ParticipantInfo]:
        """Extrai participantes da reunião usando padrões de NLP."""
        logger.info("👥 Extraindo participantes da reunião")
        
        participant_mentions = defaultdict(int)
        participant_contexts = defaultdict(list)
        
        # Busca por padrões de menção a participantes
        for pattern in self.participant_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                
                # Filtra nomes muito curtos ou comuns demais
                if len(name) < 3 or name.lower() in ['eu', 'ele', 'ela', 'você', 'nós']:
                    continue
                
                # Normaliza o nome
                name = self._normalize_name(name)
                
                if name:
                    participant_mentions[name] += 1
                    # Captura contexto ao redor da menção
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    participant_contexts[name].append(context)
        
        # Converte para lista de ParticipantInfo
        participants = []
        for name, mentions in participant_mentions.items():
            if mentions >= 1:  # Pelo menos uma menção
                # Tenta identificar função/cargo
                role = self._extract_participant_role(participant_contexts[name])
                
                # Calcula confiança baseada no número de menções e contexto
                confidence = min(0.9, 0.5 + (mentions * 0.1) + (0.2 if role else 0))
                
                if confidence >= min_confidence:
                    participants.append(ParticipantInfo(
                        name=name,
                        mentions=mentions,
                        role=role,
                        confidence=confidence
                    ))
        
        # Ordena por número de menções
        participants.sort(key=lambda p: p.mentions, reverse=True)
        
        logger.info(f"✅ {len(participants)} participantes identificados")
        return participants[:10]  # Máximo 10 participantes
    
    def _normalize_name(self, name: str) -> Optional[str]:
        """Normaliza e valida nomes de participantes."""
        # Remove artigos e preposições
        words = name.split()
        filtered_words = []
        
        for word in words:
            word_clean = word.lower().strip('.,!?:;')
            if word_clean not in ['o', 'a', 'de', 'da', 'do', 'dos', 'das', 'e']:
                if len(word_clean) >= 2 and word_clean.isalpha():
                    filtered_words.append(word.capitalize())
        
        if len(filtered_words) >= 1:
            normalized = ' '.join(filtered_words)
            # Verifica se é um nome válido (não apenas palavras comuns)
            first_name = filtered_words[0].lower()
            if first_name in self.common_names or len(filtered_words) >= 2:
                return normalized
        
        return None
    
    def _extract_participant_role(self, contexts: List[str]) -> Optional[str]:
        """Extrai função/cargo do participante baseado no contexto."""
        role_patterns = [
            r'(?:diretor|gerente|coordenador|supervisor|analista|assistente|estagiário)',
            r'(?:CEO|CTO|CFO|CMO|VP|presidente|vice)',
            r'(?:desenvolvedor|programador|designer|arquiteto|consultor)',
            r'(?:vendedor|comercial|marketing|financeiro|RH|recursos\s+humanos)'
        ]
        
        all_context = ' '.join(contexts).lower()
        
        for pattern in role_patterns:
            match = re.search(pattern, all_context, re.IGNORECASE)
            if match:
                return match.group().title()
        
        return None
    
    def _extract_main_topics(self, text: str, min_confidence: float) -> List[TopicInfo]:
        """Extrai tópicos principais da reunião."""
        logger.info("📋 Extraindo tópicos principais")
        
        # Divide o texto em segmentos/parágrafos
        segments = self._segment_text_by_topics(text)
        
        topics = []
        for i, segment in enumerate(segments):
            if len(segment.strip()) < 50:  # Segmento muito pequeno
                continue
            
            # Identifica o tópico principal do segmento
            topic_title = self._extract_topic_title(segment)
            if not topic_title:
                continue
            
            # Gera resumo do segmento
            topic_summary = self._summarize_segment(segment)
            
            # Extrai palavras-chave
            keywords = self._extract_keywords(segment)
            
            # Calcula importância baseada em palavras-chave e tamanho
            importance = self._calculate_topic_importance(segment, keywords)
            
            if importance >= 0.3:  # Filtro de relevância
                topics.append(TopicInfo(
                    title=topic_title,
                    summary=topic_summary,
                    keywords=keywords,
                    importance=importance,
                    duration_mentioned=f"~{len(segment.split())//20}min"  # Estimativa grosseira
                ))
        
        # Ordena por importância
        topics.sort(key=lambda t: t.importance, reverse=True)
        
        logger.info(f"✅ {len(topics)} tópicos principais identificados")
        return topics[:5]  # Máximo 5 tópicos principais
    
    def _segment_text_by_topics(self, text: str) -> List[str]:
        """Segmenta o texto em blocos por tópicos."""
        # Estratégia simples: divide por sentenças e agrupa por similaridade semântica
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        # Agrupa sentenças em segmentos de ~3-5 sentenças
        segments = []
        current_segment = []
        
        for sentence in sentences:
            current_segment.append(sentence)
            
            # Cria novo segmento a cada 4 sentenças ou quando detecta mudança de tópico
            if len(current_segment) >= 4 or self._detects_topic_change(sentence):
                if current_segment:
                    segments.append('. '.join(current_segment) + '.')
                    current_segment = []
        
        # Adiciona último segmento se houver
        if current_segment:
            segments.append('. '.join(current_segment) + '.')
        
        return segments
    
    def _detects_topic_change(self, sentence: str) -> bool:
        """Detecta se uma sentença indica mudança de tópico."""
        topic_change_indicators = [
            'agora vamos', 'próximo ponto', 'próximo item', 'mudando de assunto',
            'falando sobre', 'sobre o tema', 'outro tópico', 'outra questão'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in topic_change_indicators)
    
    def _extract_topic_title(self, segment: str) -> Optional[str]:
        """Extrai título do tópico de um segmento."""
        # Busca por frases que indiquem o tópico
        topic_indicators = [
            r'(?:sobre|falando\s+sobre|discutindo|apresentando)\s+([^.!?]{10,50})',
            r'(?:projeto|plano|estratégia|proposta)\s+([^.!?]{5,30})',
            r'(?:questão|problema|assunto)\s+(?:da?|do)\s+([^.!?]{5,30})'
        ]
        
        for pattern in topic_indicators:
            match = re.search(pattern, segment, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                return title.capitalize()
        
        # Fallback: usa as primeiras palavras importantes
        words = segment.split()[:10]
        important_words = [w for w in words if w.lower() in self.topic_keywords]
        
        if important_words:
            return ' '.join(important_words[:3]).capitalize()
        
        # Último recurso: primeiras palavras do segmento
        first_sentence = segment.split('.')[0]
        if len(first_sentence) > 10 and len(first_sentence) < 60:
            return first_sentence.strip().capitalize()
        
        return None
    
    def _summarize_segment(self, segment: str) -> str:
        """Gera resumo de um segmento de texto."""
        sentences = [s.strip() for s in segment.split('.') if len(s.strip()) > 15]
        
        if len(sentences) <= 2:
            return segment.strip()
        
        # Seleciona as 2 sentenças mais importantes
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Pontua por palavras-chave
            for keyword in self.topic_keywords:
                if keyword in sentence_lower:
                    score += 1
            
            # Pontua por verbos de ação
            action_verbs = ['decidiu', 'definiu', 'propôs', 'apresentou', 'discutiu']
            for verb in action_verbs:
                if verb in sentence_lower:
                    score += 1
            
            sentence_scores.append((sentence, score))
        
        # Seleciona as melhores sentenças
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:2]]
        
        return '. '.join(top_sentences) + '.'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave de um texto."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filtra palavras relevantes
        relevant_words = []
        for word in words:
            if (len(word) >= 4 and 
                word in self.topic_keywords and 
                word not in ['para', 'como', 'sobre', 'mais', 'muito', 'bem']):
                relevant_words.append(word)
        
        # Conta frequência e retorna as mais comuns
        word_counts = Counter(relevant_words)
        return [word for word, count in word_counts.most_common(5)]
    
    def _calculate_topic_importance(self, segment: str, keywords: List[str]) -> float:
        """Calcula a importância de um tópico."""
        importance = 0.0
        
        # Baseado no número de palavras-chave
        importance += len(keywords) * 0.1
        
        # Baseado no tamanho do segmento
        word_count = len(segment.split())
        if 50 <= word_count <= 200:
            importance += 0.3
        elif word_count > 200:
            importance += 0.2
        
        # Baseado na presença de verbos de ação/decisão
        action_words = ['decidiu', 'definiu', 'acordou', 'propôs', 'apresentou']
        for word in action_words:
            if word in segment.lower():
                importance += 0.2
                break
        
        return min(1.0, importance)
    
    def _extract_action_items(self, text: str, participants: List[ParticipantInfo], min_confidence: float) -> List[ActionItem]:
        """Extrai itens de ação/tarefas da reunião."""
        logger.info("📝 Extraindo itens de ação")
        
        action_items = []
        participant_names = [p.name.lower() for p in participants]
        
        for pattern in self.action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                task_text = match.group(1).strip() if len(match.groups()) >= 1 else match.group().strip()
                
                # Limpa e valida a tarefa
                task = self._clean_task_text(task_text)
                if not task or len(task) < 10:
                    continue
                
                # Tenta identificar responsável
                assignee = self._extract_assignee(match.group(), participant_names)
                
                # Tenta identificar prazo
                due_date = self._extract_due_date(text, match.start(), match.end())
                
                # Determina prioridade
                priority = self._determine_priority(task, match.group())
                
                # Calcula confiança
                confidence = self._calculate_action_confidence(task, assignee, due_date)
                
                if confidence >= min_confidence:
                    action_items.append(ActionItem(
                        task=task,
                        assignee=assignee,
                        due_date=due_date,
                        priority=priority,
                        confidence=confidence
                    ))
        
        # Remove duplicatas similares
        action_items = self._deduplicate_actions(action_items)
        
        logger.info(f"✅ {len(action_items)} itens de ação identificados")
        return action_items[:10]  # Máximo 10 ações
    
    def _clean_task_text(self, task_text: str) -> Optional[str]:
        """Limpa e normaliza o texto da tarefa."""
        # Remove palavras desnecessárias do início
        task = re.sub(r'^(que|de|para|com|por)\s+', '', task_text.strip(), flags=re.IGNORECASE)
        
        # Remove pontuação final
        task = task.rstrip('.,!?:;')
        
        # Capitaliza primeira letra
        if task and task[0].islower():
            task = task[0].upper() + task[1:]
        
        return task if len(task) >= 10 else None
    
    def _extract_assignee(self, context: str, participant_names: List[str]) -> Optional[str]:
        """Extrai responsável pela tarefa do contexto."""
        context_lower = context.lower()
        
        # Busca por nomes de participantes no contexto
        for name in participant_names:
            if name in context_lower:
                return name.title()
        
        # Busca por pronomes que indicam responsável
        if re.search(r'\b(eu|vou|farei)\b', context_lower):
            return "Falante atual"
        
        return None
    
    def _extract_due_date(self, text: str, start_pos: int, end_pos: int) -> Optional[str]:
        """Extrai prazo mencionado próximo à tarefa."""
        # Busca em um contexto de 100 caracteres ao redor da tarefa
        context_start = max(0, start_pos - 100)
        context_end = min(len(text), end_pos + 100)
        context = text[context_start:context_end]
        
        for pattern in self.deadline_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group().strip()
        
        return None
    
    def _determine_priority(self, task: str, context: str) -> Priority:
        """Determina prioridade da tarefa baseada no contexto."""
        high_priority_words = ['urgente', 'imediato', 'crítico', 'importante', 'prioridade']
        low_priority_words = ['quando possível', 'sem pressa', 'eventualmente']
        
        context_lower = (task + ' ' + context).lower()
        
        if any(word in context_lower for word in high_priority_words):
            return Priority.ALTA
        elif any(word in context_lower for word in low_priority_words):
            return Priority.BAIXA
        else:
            return Priority.MEDIA
    
    def _calculate_action_confidence(self, task: str, assignee: Optional[str], due_date: Optional[str]) -> float:
        """Calcula confiança na extração de uma ação."""
        confidence = 0.5  # Base
        
        # Aumenta confiança se tem responsável
        if assignee:
            confidence += 0.2
        
        # Aumenta confiança se tem prazo
        if due_date:
            confidence += 0.2
        
        # Aumenta confiança se tem verbos de ação claros
        action_verbs = ['fazer', 'executar', 'entregar', 'enviar', 'preparar', 'revisar']
        if any(verb in task.lower() for verb in action_verbs):
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _deduplicate_actions(self, actions: List[ActionItem]) -> List[ActionItem]:
        """Remove ações duplicadas ou muito similares."""
        unique_actions = []
        
        for action in actions:
            is_duplicate = False
            for existing in unique_actions:
                # Verifica similaridade simples baseada nas primeiras palavras
                action_words = set(action.task.lower().split()[:5])
                existing_words = set(existing.task.lower().split()[:5])
                
                # Se mais de 60% das palavras são iguais, considera duplicata
                if len(action_words & existing_words) / len(action_words | existing_words) > 0.6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_actions.append(action)
        
        return unique_actions
    
    def _extract_key_decisions(self, text: str, min_confidence: float) -> List[KeyDecision]:
        """Extrai decisões importantes da reunião."""
        logger.info("⚖️ Extraindo decisões importantes")
        
        decisions = []
        
        for pattern in self.decision_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                decision_text = match.group(1).strip() if len(match.groups()) >= 1 else match.group().strip()
                
                # Limpa o texto da decisão
                decision = self._clean_decision_text(decision_text)
                if not decision or len(decision) < 15:
                    continue
                
                # Extrai contexto
                context = self._extract_decision_context(text, match.start(), match.end())
                
                # Determina impacto
                impact = self._determine_decision_impact(decision, context)
                
                # Calcula confiança
                confidence = self._calculate_decision_confidence(decision, context)
                
                if confidence >= min_confidence:
                    decisions.append(KeyDecision(
                        decision=decision,
                        context=context,
                        impact=impact,
                        confidence=confidence
                    ))
        
        # Remove duplicatas
        decisions = self._deduplicate_decisions(decisions)
        
        logger.info(f"✅ {len(decisions)} decisões importantes identificadas")
        return decisions[:5]  # Máximo 5 decisões
    
    def _clean_decision_text(self, decision_text: str) -> Optional[str]:
        """Limpa e normaliza o texto da decisão."""
        decision = decision_text.strip().rstrip('.,!?:;')
        
        if decision and decision[0].islower():
            decision = decision[0].upper() + decision[1:]
        
        return decision if len(decision) >= 15 else None
    
    def _extract_decision_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extrai contexto ao redor de uma decisão."""
        context_start = max(0, start_pos - 150)
        context_end = min(len(text), end_pos + 150)
        context = text[context_start:context_end].strip()
        
        # Pega a sentença completa
        sentences = re.split(r'[.!?]+', context)
        if len(sentences) >= 3:
            return sentences[len(sentences)//2].strip()
        
        return context[:200] + "..." if len(context) > 200 else context
    
    def _determine_decision_impact(self, decision: str, context: str) -> str:
        """Determina o impacto de uma decisão."""
        high_impact_words = ['estratégico', 'importante', 'crítico', 'fundamental', 'grande']
        low_impact_words = ['pequeno', 'simples', 'menor', 'básico']
        
        combined_text = (decision + ' ' + context).lower()
        
        if any(word in combined_text for word in high_impact_words):
            return 'alta'
        elif any(word in combined_text for word in low_impact_words):
            return 'baixa'
        else:
            return 'média'
    
    def _calculate_decision_confidence(self, decision: str, context: str) -> float:
        """Calcula confiança na extração de uma decisão."""
        confidence = 0.6  # Base
        
        # Aumenta se tem palavras de decisão claras
        decision_words = ['decidiu', 'aprovado', 'definido', 'acordado', 'resolvido']
        if any(word in (decision + context).lower() for word in decision_words):
            confidence += 0.2
        
        # Aumenta se tem contexto substancial
        if len(context) > 50:
            confidence += 0.1
        
        return min(0.9, confidence)
    
    def _deduplicate_decisions(self, decisions: List[KeyDecision]) -> List[KeyDecision]:
        """Remove decisões duplicadas."""
        unique_decisions = []
        
        for decision in decisions:
            is_duplicate = False
            for existing in unique_decisions:
                # Verifica similaridade
                decision_words = set(decision.decision.lower().split()[:7])
                existing_words = set(existing.decision.lower().split()[:7])
                
                if len(decision_words & existing_words) / len(decision_words | existing_words) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_decisions.append(decision)
        
        return unique_decisions
    
    def _analyze_sentiment(self, text: str, topics: List[TopicInfo]) -> SentimentAnalysis:
        """Realiza análise de sentimento do texto."""
        logger.info("😊 Analisando sentimento da reunião")
        
        # Análise geral
        overall_sentiment = self._calculate_overall_sentiment(text)
        
        # Análise por tópico
        topic_sentiments = {}
        for topic in topics:
            topic_sentiment = self._calculate_topic_sentiment(topic.summary)
            topic_sentiments[topic.title] = topic_sentiment
        
        return SentimentAnalysis(
            overall=overall_sentiment,
            topics=topic_sentiments,
            confidence=0.7  # Análise de sentimento simples tem confiança moderada
        )
    
    def _calculate_overall_sentiment(self, text: str) -> str:
        """Calcula sentimento geral do texto."""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count > negative_count * 1.5:
            return "positivo"
        elif negative_count > positive_count * 1.5:
            return "negativo"
        else:
            return "neutro"
    
    def _calculate_topic_sentiment(self, topic_text: str) -> str:
        """Calcula sentimento de um tópico específico."""
        return self._calculate_overall_sentiment(topic_text)
    
    def _generate_structured_summary(
        self, 
        text: str, 
        participants: List[ParticipantInfo], 
        topics: List[TopicInfo], 
        actions: List[ActionItem], 
        decisions: List[KeyDecision]
    ) -> str:
        """Gera resumo estruturado da reunião."""
        summary_parts = []
        
        # Cabeçalho
        summary_parts.append("## RESUMO DA REUNIÃO")
        summary_parts.append("")
        
        # Participantes
        if participants:
            summary_parts.append("### 👥 Participantes:")
            for p in participants[:5]:
                role_text = f" ({p.role})" if p.role else ""
                summary_parts.append(f"- {p.name}{role_text}")
            summary_parts.append("")
        
        # Tópicos principais
        if topics:
            summary_parts.append("### 📋 Tópicos Principais:")
            for i, topic in enumerate(topics[:3], 1):
                summary_parts.append(f"{i}. **{topic.title}**")
                summary_parts.append(f"   {topic.summary}")
            summary_parts.append("")
        
        # Decisões importantes
        if decisions:
            summary_parts.append("### ⚖️ Decisões Importantes:")
            for i, decision in enumerate(decisions[:3], 1):
                summary_parts.append(f"{i}. {decision.decision}")
            summary_parts.append("")
        
        # Itens de ação
        if actions:
            summary_parts.append("### 📝 Próximas Ações:")
            for i, action in enumerate(actions[:5], 1):
                assignee_text = f" (Responsável: {action.assignee})" if action.assignee else ""
                due_text = f" - Prazo: {action.due_date}" if action.due_date else ""
                priority_emoji = {"alta": "🔴", "média": "🟡", "baixa": "🟢"}.get(action.priority.value, "🟡")
                summary_parts.append(f"{i}. {priority_emoji} {action.task}{assignee_text}{due_text}")
            summary_parts.append("")
        
        # Resumo geral
        summary_parts.append("### 📄 Resumo Geral:")
        general_summary = self._generate_general_summary(text)
        summary_parts.append(general_summary)
        
        return "\n".join(summary_parts)
    
    def _generate_general_summary(self, text: str) -> str:
        """Gera resumo geral do texto."""
        # Estratégia simples: pega as 3 sentenças mais importantes
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        if len(sentences) <= 3:
            return '. '.join(sentences) + '.'
        
        # Pontua sentenças por importância
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Palavras-chave importantes
            for keyword in self.topic_keywords:
                if keyword in sentence_lower:
                    score += 1
            
            # Verbos de ação/decisão
            action_words = ['decidiu', 'definiu', 'discutiu', 'apresentou', 'propôs']
            for word in action_words:
                if word in sentence_lower:
                    score += 2
            
            sentence_scores.append((sentence, score))
        
        # Seleciona as melhores
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:3]]
        
        return '. '.join(top_sentences) + '.'
    
    def _calculate_overall_confidence(
        self, 
        participants: List[ParticipantInfo], 
        topics: List[TopicInfo], 
        actions: List[ActionItem], 
        decisions: List[KeyDecision]
    ) -> float:
        """Calcula confiança geral da análise."""
        confidences = []
        
        # Confiança dos participantes
        if participants:
            avg_participant_confidence = sum(p.confidence for p in participants) / len(participants)
            confidences.append(avg_participant_confidence)
        
        # Confiança dos tópicos (baseada na importância)
        if topics:
            avg_topic_confidence = sum(t.importance for t in topics) / len(topics)
            confidences.append(avg_topic_confidence)
        
        # Confiança das ações
        if actions:
            avg_action_confidence = sum(a.confidence for a in actions) / len(actions)
            confidences.append(avg_action_confidence)
        
        # Confiança das decisões
        if decisions:
            avg_decision_confidence = sum(d.confidence for d in decisions) / len(decisions)
            confidences.append(avg_decision_confidence)
        
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.5  # Confiança padrão se não há dados


# Instância global do serviço
meeting_analysis_service = MeetingAnalysisService() 