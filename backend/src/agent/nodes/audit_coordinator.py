import os
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import OverallState
from agent.configuration import Configuration
from agent.tools_and_schemas import (
    DocumentReaderTool, 
    AnswerSaverTool
)


def get_audit_status():
    """Get current audit questionnaire status"""
    result = DocumentReaderTool.read_document("cuestionario_auditoria_nes.md")
    if not result["success"]:
        error_msg = f"Error loading questionnaire: {result['error']}"
        return error_msg, []
    
    questions = AnswerSaverTool.load_answers(result["questions"])
    
    # Format questions with status
    questions_text = "CUESTIONARIO DE AUDITORÍA NES:\n\n"
    for q in questions:
        status_emoji = "✅" if q.status == "answered" else "⏳"
        questions_text += f"{status_emoji} {q.title}: {q.content}\n"
        if q.status == "answered" and q.answer:
            questions_text += f"   RESPUESTA ACTUAL: {q.answer[:100]}...\n"
        questions_text += "\n"
    
    # Add progress summary
    total = len(questions)
    answered = len([q for q in questions if q.status == "answered"])
    if total > 0:
        questions_text += f"PROGRESO: {answered}/{total} preguntas completadas ({answered/total*100:.1f}%)"
    else:
        questions_text += "PROGRESO: No hay preguntas disponibles"
    
    return questions_text, questions


def save_user_answer(question_id: str, answer: str):
    """Save an answer for a specific audit question"""
    # Load questions to get the full list
    result = DocumentReaderTool.read_document("cuestionario_auditoria_nes.md")
    if not result["success"]:
        return f"Error: Could not load questions to save answer"
    
    questions = result["questions"]
    save_result = AnswerSaverTool.save_answer(question_id, answer, questions)
    
    if save_result["success"]:
        return "✅ Respuesta guardada exitosamente para " + question_id
    else:
        return "❌ Error guardando respuesta: " + save_result['error']


def audit_coordinator_agent(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Enhanced single coordinator agent using comprehensive prompt with embedded expertise.
    Simple approach: All logic in one comprehensive prompt, tools called as needed.
    """
    
    # Get configuration 
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=configurable.reflection_model,
        temperature=0.1,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Get user message - handle LangChain message objects
    messages = state.get("messages", [])
    user_messages = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == "human":
            user_messages.append(msg)
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_messages.append(msg)
    
    if user_messages:
        latest_msg = user_messages[-1]
        if hasattr(latest_msg, 'content'):
            latest_user_message = latest_msg.content
        else:
            latest_user_message = latest_msg.get("content", "Hola")
    else:
        latest_user_message = "Hola"
    
    # Get current audit status
    questions_status, questions_list = get_audit_status()
    
    # Build comprehensive prompt with embedded NES expertise and current status
    prompt = f"""Eres un asistente experto en auditorías de seguridad según el estándar NES (Esquema Nacional de Seguridad) de España.

ESTADO ACTUAL DEL CUESTIONARIO:
{questions_status}

CONOCIMIENTO NES PARA EVALUAR RESPUESTAS:
- Copias de seguridad: Requiere tipo de sistema, frecuencia, verificación, ubicación (local/remota), retención, plan de recuperación
- Control de acceso: Autenticación implementada, políticas de contraseñas, MFA, gestión de privilegios, revisiones periódicas, logs
- Monitoreo: Herramientas de red, sistemas de detección, análisis de logs, procedimientos de respuesta, escalación, informes
- Continuidad: Plan documentado, procedimientos de recuperación, RTO definido, pruebas regulares, documentación actualizada
- Formación: Contenidos específicos, frecuencia, evaluación de efectividad, registros de cumplimiento, certificaciones
- Vulnerabilidades: Herramientas de análisis, frecuencia de evaluaciones, procedimientos de parcheo, tiempos de respuesta
- Cifrado: Algoritmos utilizados, gestión de claves, datos en tránsito/reposo, cumplimiento RGPD, clasificación de datos
- Auditorías internas: Frecuencia, alcance, personal responsable, seguimiento de hallazgos, documentación de evidencias

INSTRUCCIONES:
1. SIEMPRE responde en español, tono conversacional y profesional
2. Si es primer saludo: Da bienvenida y muestra primera pregunta pendiente  
3. Si usuario responde a pregunta: Evalúa completitud contra requisitos NES arriba
4. Si respuesta completa: Di que la guardarás y muestra siguiente pregunta
5. Si respuesta incompleta: Pide específicamente qué falta según NES
6. Si pide progreso: Resumen desde estado actual arriba
7. Siempre sugiere siguiente paso claro

MENSAJE DEL USUARIO: {latest_user_message}

Analiza el mensaje y responde apropiadamente. Si el usuario está proporcionando una respuesta que parece completa según los estándares NES, menciona que la guardarás (pero no hagas la acción aún en esta respuesta)."""

    # Get response from LLM
    response = llm.invoke(prompt)
    
    # Check if we need to save an answer based on the response
    # Simple heuristic: if response mentions saving/guardando, try to save
    response_content = response.content.lower()
    should_save = any(keyword in response_content for keyword in ["guardar", "guardo", "guardado", "salvar"])
    
    # If we should save and user message looks like substantial answer
    if should_save and len(latest_user_message.strip()) > 20:
        # Try to determine which question they're answering
        # Simple approach: find first pending question
        for question in questions_list:
            if question.status == "pending":
                save_result = save_user_answer(question.id, latest_user_message)
                if "guardada exitosamente" in save_result:
                    response.content += f"\n\n✅ {save_result}"
                break
    
    # Update conversation history
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "user": latest_user_message,
        "assistant": response.content,
        "timestamp": "current"
    })
    
    # Return updated state with all required fields
    return {
        "messages": state.get("messages", []) + [{
            "role": "assistant",
            "content": response.content
        }],
        "document_path": "cuestionario_auditoria_nes.md",
        "conversation_history": conversation_history,
        "questions_status": {q.id: q.status for q in questions_list} if questions_list else {},
        "current_question": next((q.id for q in questions_list if q.status == "pending"), None) if questions_list else None,
        "user_context": state.get("user_context", {}),
        "language": "es"
    }