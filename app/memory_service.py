"""
Serviço de memória conversacional.
- Extrai fatos/informações importantes de cada conversa e salva no SQLite.
- Busca memórias relevantes para enriquecer o contexto do chat.
- Busca histórico de conversas anteriores por similaridade.
"""

from datetime import datetime
from typing import List, Tuple

from app.database import (
    ChatMemory, ChatMessage, ChatSession, SessionLocal,
)


# ── Palavras indicadoras de informação importante ────────────────
_FACT_INDICATORS = [
    "meu nome é", "me chamo", "eu sou", "trabalho com", "trabalho na",
    "meu ip é", "o ip é", "o endereço é", "a senha é", "o servidor é",
    "o modelo é", "a porta é", "estou usando", "nós usamos", "a empresa",
    "o cliente", "o contrato", "o número", "sempre que", "lembre",
    "anota", "guarda isso", "não esqueça", "importante:",
    "meu email", "meu telefone", "o problema é", "a solução é",
    "a rede", "a vlan", "o switch", "o roteador", "o olt",
    "configuração", "topologia", "endereço", "equipamento",
]

# Frases que indicam pedido de memória ("lembre", "guarde", etc.)
_EXPLICIT_MEMORY = [
    "lembre", "lembra", "anota", "guarda", "guarde", "memorize",
    "não esqueça", "nao esqueca", "salva isso", "salve isso",
    "registra isso", "importante:", "nota:",
]
# Padrões para nomes (palavras maiúsculas que viçam depois de "sou", "chamo", etc)
_NAME_PATTERNS = ['chamo', 'sou', 'nome é', 'meu nome', 'é ', 'chamado']

def extract_facts_from_exchange(
    user_msg: str,
    assistant_msg: str,
    session_id: str = "",
) -> List[str]:
    """
    Analisa uma troca user↔assistant e extrai fatos relevantes
    para salvar na memória de longo prazo.
    
    Prioriza NOMES para garantir que sejam salvos.
    """
    facts: List[str] = []
    user_lower = user_msg.lower()

    # 1) DETECTAR NOMES ESPECIFICAMENTE
    # Procurar padrões: "meu nome é X", "me chamo X", "sou X"
    name_indicators = [
        'meu nome é', 'me chamo', 'chamo-me', 'me chama', 'me chama de',
        'eu sou ', 'meu nome:', 'nome é', 'meu nome é:',
    ]
    for indicator in name_indicators:
        if indicator in user_lower:
            # Extrair o que vem depois do indicador
            idx = user_lower.find(indicator)
            after_indicator = user_msg[idx + len(indicator):].strip()
            # Pegar a primeira "coisa" após o indicador
            potential_name = after_indicator.split()[0] if after_indicator.split() else ""
            if potential_name and len(potential_name) > 1:
                # Salvar de forma que seja fácil encontrar depois
                facts.append(f"NOME: {potential_name.strip('.,!?')}")
                facts.append(f"O nome do usuário é {potential_name.strip('.,!?')}")
                return facts  # Nome é prioridade máxima

    # 2) O usuário pediu explicitamente para lembrar algo?
    for trigger in _EXPLICIT_MEMORY:
        if trigger in user_lower:
            facts.append(f"O usuário pediu para lembrar: {user_msg.strip()}")
            return facts  # Não precisa analisar mais

    # 3) A mensagem contém indicadores de informação útil?
    matched_indicators = []
    for indicator in _FACT_INDICATORS:
        if indicator in user_lower:
            matched_indicators.append(indicator)

    if matched_indicators:
        # Extrair frases que contêm os indicadores
        sentences = [s.strip() for s in user_msg.replace("\n", ". ").split(".") if s.strip()]
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator in matched_indicators:
                if indicator in sentence_lower and len(sentence.strip()) > 5:
                    facts.append(sentence.strip())
                    break  # Evitar duplicar mesma frase

    # 4) Se não encontrou fatos com indicadores, mas a mensagem é informativa, salvar como contexto
    if not facts:
        is_question = user_lower.strip().endswith("?")
        # Mensagens que não são só perguntas simples devem ser salvas
        if not is_question and len(user_msg.strip()) > 10:
            # Salvar a mensagem inteira como contexto importante
            facts.append(f"O usuário disse: {user_msg.strip()}")
        elif len(user_msg.strip()) > 50:
            # Mesmo que seja pergunta, salvar se for longa (pode conter contexto importante)
            facts.append(f"O usuário perguntou: {user_msg.strip()}")

    return facts[:10]  # Máximo de 10 fatos por troca


def store_facts(facts: List[str], session_id: str = "", category: str = "general"):
    """Salva fatos extraídos na tabela chat_memories."""
    if not facts:
        return

    db = SessionLocal()
    try:
        for fact in facts:
            if len(fact.strip()) < 5:
                continue
            # Evitar duplicatas exatas
            existing = db.query(ChatMemory).filter(ChatMemory.fact == fact.strip()).first()
            if existing:
                continue
            db.add(ChatMemory(
                fact=fact.strip(),
                source_session_id=session_id,
                category=category,
            ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def get_relevant_memories(query: str, max_results: int = 5) -> List[str]:
    """
    Busca memórias (fatos de conversas anteriores) relevantes para a query.
    Usa correspondência de palavras-chave e sem semântica inteligente.
    """
    db = SessionLocal()
    try:
        all_memories = db.query(ChatMemory).order_by(ChatMemory.created_at.desc()).all()
        if not all_memories:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Palavras-chave para diferentes tipos de memória
        personal_keywords = {
            'nome': ['nome:', 'nome', 'chamo', 'sou', 'chamado', 'meu nome', 'me chama'],
            'email': ['email', 'mail', '@'],
            'telefone': ['telefone', 'whatsapp', 'celular', 'fone', 'número'],
            'trabalho': ['trabalho', 'empresa', 'cargo', 'função', 'profissão'],
            'localização': ['mora', 'cidade', 'estado', 'endereço', 'local'],
        }
        
        # Verificar que tipo de informação está sendo pedida
        query_type = 'general'
        for info_type, keywords in personal_keywords.items():
            if any(kw in query_lower for kw in keywords):
                query_type = info_type
                break
        
        scored = []
        
        for mem in all_memories:
            mem_fact = mem.fact.lower()
            
            # Se está buscando informação pessoal, priorize memórias sobre isso
            if query_type in personal_keywords:
                type_keywords = personal_keywords[query_type]
                # CRÍTICO: Procurar não só as palavras-chave, mas também o padrão "NOME:"
                if query_type == 'nome':
                    # Procurar explicitamente por "NOME:" ou "nome:"
                    if 'nome:' in mem_fact:
                        scored.append((500, mem.fact))  # Score MÁXIMO
                        continue
                
                # Também procurar por keywords gerais
                if any(kw in mem_fact for kw in type_keywords):
                    # Encontrou memória sobre o tipo específico
                    scored.append((100, mem.fact))  # Score muito alto
                    continue
            
            # Busca geral por palavras-chave
            mem_words = set(mem_fact.split())
            score = len(query_words.intersection(mem_words))
            if score > 0:
                scored.append((score, mem.fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [fact for _, fact in scored[:max_results]]
        return result
    except Exception as e:
        print(f"[ERROR] get_relevant_memories falhou: {e}")
        return []
    finally:
        db.close()


def get_past_conversations_context(
    query: str,
    current_session_id: str = "",
    max_sessions: int = 3,
    max_messages_per_session: int = 4,
) -> str:
    """
    Busca conversas ANTERIORES relevantes para a query atual.
    Retorna um resumo formatado das conversas passadas relacionadas.
    
    NÃO traz conversas antigas quando a pergunta é sobre informações pessoais.
    """
    db = SessionLocal()
    try:
        # Se a pergunta é sobre informações pessoais, não trazer conversas antigas
        personal_keywords = ['nome', 'email', 'telefone', 'chamo', 'sou', 'trabalho', 'empresa']
        query_lower = query.lower()
        
        # Checar se é pergunta pessoal
        is_personal_query = any(kw in query_lower for kw in personal_keywords)
        
        # Se é pergunta pessoal, não trazer conversas antigas - só usar memórias diretas
        if is_personal_query:
            return ""  # Deixar para as memórias salvas responderem
        
        # Buscar todas as sessões que NÃO são a sessão atual
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.id != current_session_id)
            .order_by(ChatSession.created_at.desc())
            .limit(50)  # Buscar nas últimas 50 sessões
            .all()
        )

        if not sessions:
            return ""

        query_words = set(query_lower.split())
        scored_sessions: List[Tuple[int, str, list]] = []

        for sess in sessions:
            # Buscar mensagens da sessão
            msgs = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == sess.id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            if not msgs:
                continue

            # Pontuar pela relevância das mensagens
            session_text = " ".join(m.content.lower() for m in msgs)
            session_words = set(session_text.split())
            score = len(query_words.intersection(session_words))

            if score > 0:
                msg_data = [(m.role, m.content) for m in msgs]
                scored_sessions.append((score, sess.id, msg_data))

        scored_sessions.sort(key=lambda x: x[0], reverse=True)

        # Formatar contexto das melhores sessões
        parts = []
        for _, sess_id, msgs in scored_sessions[:max_sessions]:
            conv_lines = []
            for role, content in msgs[-max_messages_per_session:]:
                prefix = "Usuário" if role == "user" else "Assistente"
                # Limitar tamanho do conteúdo
                short = content[:300] + "..." if len(content) > 300 else content
                conv_lines.append(f"  {prefix}: {short}")

            if conv_lines:
                parts.append("Conversa anterior:\n" + "\n".join(conv_lines))

        return "\n\n".join(parts)
    except Exception:
        return ""
    finally:
        db.close()


def get_recent_memories_summary(limit: int = 10) -> str:
    """Retorna um resumo das memórias mais recentes."""
    db = SessionLocal()
    try:
        memories = (
            db.query(ChatMemory)
            .order_by(ChatMemory.created_at.desc())
            .limit(limit)
            .all()
        )
        if not memories:
            return ""
        return "\n".join(f"- {m.fact}" for m in memories)
    except Exception:
        return ""
    finally:
        db.close()
