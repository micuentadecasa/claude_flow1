import os
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import OverallState
from agent.configuration import Configuration
from agent.tools_and_schemas import (
    DocumentReaderTool, 
    QuestionParserTool, 
    AnswerEnhancementTool, 
    ProgressTrackerTool,
    AnswerSaverTool,
    AuditQuestion
)


def audit_coordinator_agent(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Single coordinator agent that handles all audit interactions in Spanish.
    
    Responsibilities:
    1. Document reading and parsing
    2. Progress tracking and status reporting
    3. Interactive conversational guidance
    4. Answer enhancement with NES expertise
    5. Natural Spanish language responses
    """
    
    # Get configuration (TIP #012)
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize LLM with configured model
    llm = ChatGoogleGenerativeAI(
        model=configurable.reflection_model,
        temperature=0.1,  # Low temperature for consistent guidance
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Initialize state if needed
    if not state.get("document_path"):
        state["document_path"] = "cuestionario_auditoria_nes.md"
        state["language"] = "es"
        state["user_context"] = {}
        state["conversation_history"] = []
        state["questions_status"] = {}
    
    # Load audit document
    document_result = DocumentReaderTool.read_document(state["document_path"])
    
    if not document_result["success"]:
        return {
            "messages": [{
                "role": "assistant",
                "content": f"Lo siento, no pude cargar el cuestionario de auditoría. Error: {document_result['error']}"
            }]
        }
    
    questions = document_result["questions"]
    
    # Load saved answers from file
    questions = AnswerSaverTool.load_answers(questions)
    
    # Update questions status from state
    for question in questions:
        if question.id in state["questions_status"]:
            question.status = state["questions_status"][question.id]
    
    # Get latest user message
    user_messages = []
    for msg in state["messages"]:
        if hasattr(msg, 'content') and hasattr(msg, '__class__') and 'HumanMessage' in str(type(msg)):
            # LangChain HumanMessage object
            user_messages.append(msg)
        elif isinstance(msg, dict) and msg.get("role") == "user":
            # Dictionary message
            user_messages.append(msg)
    
    if user_messages:
        last_msg = user_messages[-1]
        if hasattr(last_msg, 'content'):
            latest_user_message = last_msg.content
        elif isinstance(last_msg, dict):
            latest_user_message = last_msg.get("content", "")
        else:
            latest_user_message = str(last_msg)
    else:
        latest_user_message = ""
    
    # Determine conversation context and intent
    conversation_context = _analyze_user_intent(latest_user_message, questions, state)
    
    # Generate Spanish response based on context
    response_content = _generate_spanish_response(
        llm, conversation_context, questions, state, latest_user_message
    )
    
    # Update state based on interaction
    updated_state = _update_state_after_interaction(
        state, conversation_context, questions, latest_user_message
    )
    
    # Add assistant response to messages
    updated_state["messages"] = state["messages"] + [{
        "role": "assistant", 
        "content": response_content
    }]
    
    return updated_state


def _analyze_user_intent(user_message: str, questions: list, state: OverallState) -> Dict[str, Any]:
    """Analyze user message to determine intent and context"""
    
    user_lower = user_message.lower()
    
    # Check for status/progress requests
    if any(keyword in user_lower for keyword in ["estado", "progreso", "cuántas", "faltan"]):
        return {"intent": "status_check", "type": "progress"}
    
    # Check for next question requests
    if any(keyword in user_lower for keyword in ["siguiente", "próxima", "continuar"]):
        return {"intent": "next_question", "type": "navigation"}
    
    # Check for help/start requests
    if any(keyword in user_lower for keyword in ["ayuda", "empezar", "iniciar", "hola"]):
        return {"intent": "help_start", "type": "greeting"}
    
    # Check if user is answering a question
    current_question_id = state.get("current_question")
    if current_question_id and len(user_message.strip()) > 10:
        return {
            "intent": "answer_question", 
            "type": "answer",
            "question_id": current_question_id
        }
    
    # If no current question set, check if this looks like an answer to the first pending question
    if len(user_message.strip()) > 15:  # Substantial message
        next_question = QuestionParserTool.get_next_question(questions)
        if next_question:
            return {
                "intent": "answer_question",
                "type": "answer", 
                "question_id": next_question.id
            }
    
    # Default to conversational assistance
    return {"intent": "general_assistance", "type": "conversation"}


def _generate_spanish_response(
    llm: ChatGoogleGenerativeAI, 
    context: Dict[str, Any], 
    questions: list, 
    state: OverallState,
    user_message: str
) -> str:
    """Generate appropriate Spanish response based on context"""
    
    intent = context["intent"]
    
    if intent == "status_check":
        return _generate_progress_response(questions)
    
    elif intent == "next_question":
        return _generate_next_question_response(questions, state)
    
    elif intent == "help_start":
        return _generate_welcome_response(questions)
    
    elif intent == "answer_question":
        return _generate_answer_enhancement_response(
            llm, context["question_id"], user_message, questions, state
        )
    
    else:
        return _generate_general_assistance_response(llm, user_message, questions, state)


def _generate_progress_response(questions: list) -> str:
    """Generate progress status response in Spanish"""
    progress_summary = ProgressTrackerTool.get_completion_summary(questions)
    next_action = ProgressTrackerTool.suggest_next_action(questions)
    
    return f"""
{progress_summary}

{next_action}

¿Te gustaría continuar con la siguiente pregunta o prefieres revisar alguna respuesta anterior?
    """.strip()


def _generate_next_question_response(questions: list, state: OverallState) -> str:
    """Generate next question response in Spanish"""
    next_question = QuestionParserTool.get_next_question(questions)
    
    if not next_question:
        return "¡Excelente! Has completado todas las preguntas de la auditoría de seguridad NES. ¿Te gustaría revisar alguna respuesta o generar un resumen final?"
    
    return f"""
**{next_question.title}**

{next_question.content}

Por favor, proporciona tu respuesta considerando los estándares de seguridad NES. Incluye todos los detalles relevantes que puedas.
    """.strip()


def _generate_welcome_response(questions: list) -> str:
    """Generate welcome/help response in Spanish"""
    total_questions = len(questions)
    answered_count = len([q for q in questions if q.status == "answered"])
    next_question = QuestionParserTool.get_next_question(questions)
    
    welcome_text = f"""
¡Hola! Soy tu asistente especializado en auditorías de seguridad según el estándar NES de España.

📋 **Estado del Cuestionario:**
- Total de preguntas: {total_questions}
- Completadas: {answered_count}
- Pendientes: {total_questions - answered_count}
- Progreso: {(answered_count/total_questions*100):.1f}%

🎯 **Te ayudo a:**
- Responder las preguntas paso a paso con los detalles requeridos por NES
- Mejorar tus respuestas para cumplir con los estándares
- Guardar automáticamente tus respuestas
- Guiarte hasta completar toda la auditoría
    """
    
    if next_question:
        welcome_text += f"""

🚀 **Empecemos con la siguiente pregunta:**

**{next_question.title}**

{next_question.content}

Por favor, proporciona tu respuesta con el mayor detalle posible. Te ayudaré a asegurar que cumple con todos los requisitos NES.
        """
    else:
        welcome_text += """

🎉 **¡Felicitaciones!** Ya has completado todas las preguntas de la auditoría.
        """
    
    return welcome_text.strip()


def _generate_answer_enhancement_response(
    llm: ChatGoogleGenerativeAI,
    question_id: str, 
    user_answer: str,
    questions: list,
    state: OverallState
) -> str:
    """Generate answer enhancement response using NES expertise"""
    
    # Find the question
    current_question = None
    for q in questions:
        if q.id == question_id:
            current_question = q
            break
    
    if not current_question:
        return "No pude encontrar la pregunta actual. ¿Podrías repetir tu respuesta?"
    
    # Enhance answer using NES expertise
    enhancement = AnswerEnhancementTool.enhance_answer(
        current_question.content, user_answer
    )
    
    if not enhancement["enhancement_needed"]:
        # Answer is sufficient - SAVE IT!
        save_result = AnswerSaverTool.save_answer(question_id, user_answer, questions)
        
        if save_result["success"]:
            # Find next question
            next_question = QuestionParserTool.get_next_question(questions)
            
            if next_question:
                return f"""
✅ ¡Excelente respuesta! Tu explicación cubre adecuadamente los requisitos del estándar NES.

📝 **He guardado tu respuesta para:** {current_question.title}

🔄 **Siguiente pregunta:**

**{next_question.title}**

{next_question.content}

Por favor, proporciona tu respuesta considerando los estándares de seguridad NES.
                """.strip()
            else:
                return f"""
✅ ¡Excelente respuesta! Tu explicación cubre adecuadamente los requisitos del estándar NES.

📝 **He guardado tu respuesta para:** {current_question.title}

🎉 **¡Felicitaciones!** Has completado todas las preguntas de la auditoría de seguridad NES. 

¿Te gustaría revisar alguna respuesta o generar un resumen de la auditoría?
                """.strip()
        else:
            return f"Tu respuesta es excelente, pero hubo un error al guardarla: {save_result['error']}. ¿Podrías repetirla?"
    
    else:
        # Answer needs improvement - ask for specific details
        missing_requirements = enhancement["missing_requirements"]
        
        prompt = f"""
Como experto en auditorías de seguridad NES de España, ayuda al usuario a completar su respuesta.

Pregunta actual: {current_question.content}
Respuesta del usuario: {user_answer}

Información adicional requerida según NES:
{chr(10).join([f"- {req}" for req in missing_requirements])}

Genera una respuesta en español que:
1. Reconozca positivamente lo que el usuario ya proporcionó
2. Explique específicamente qué información adicional necesitas
3. Haga 2-3 preguntas concretas para obtener esos detalles
4. Mantenga un tono alentador y profesional
5. Termine pidiendo que complemente su respuesta

NO guardes la respuesta aún, necesitamos más información.
        """
        
        response = llm.invoke(prompt)
        return response.content


def _generate_general_assistance_response(
    llm: ChatGoogleGenerativeAI,
    user_message: str,
    questions: list,
    state: OverallState
) -> str:
    """Generate general conversational assistance response"""
    
    progress = QuestionParserTool.get_question_status(questions)
    
    prompt = f"""
Eres un asistente experto en auditorías de seguridad según el estándar NES de España.

Contexto actual:
- Progreso de auditoría: {progress.answered}/{progress.total_questions} preguntas completadas
- Mensaje del usuario: {user_message}

Proporciona una respuesta útil en español que:
1. Responda a la consulta del usuario
2. Mantenga el foco en la auditoría de seguridad
3. Ofrezca orientación práctica
4. Sea conversacional y profesional

Si el usuario pregunta sobre temas fuera de la auditoría, redirige amablemente hacia el cuestionario.
    """
    
    response = llm.invoke(prompt)
    return response.content


def _update_state_after_interaction(
    state: OverallState, 
    context: Dict[str, Any],
    questions: list,
    user_message: str
) -> OverallState:
    """Update state based on the interaction"""
    
    updated_state = state.copy()
    
    # Update current question based on intent
    if context["intent"] == "next_question":
        next_question = QuestionParserTool.get_next_question(questions)
        if next_question:
            updated_state["current_question"] = next_question.id
    
    elif context["intent"] == "answer_question":
        question_id = context["question_id"]
        
        # Check if answer was complete and saved
        current_question = None
        for q in questions:
            if q.id == question_id:
                current_question = q
                break
        
        if current_question and current_question.status == "answered":
            # Answer was saved, mark in state and move to next
            updated_state["questions_status"][question_id] = "answered"
            
            # Move to next question
            next_question = QuestionParserTool.get_next_question(questions)
            if next_question:
                updated_state["current_question"] = next_question.id
            else:
                updated_state["current_question"] = None
        else:
            # Answer needs more work, keep current question
            updated_state["current_question"] = question_id
    
    elif context["intent"] == "help_start":
        # Set current question to first pending question
        next_question = QuestionParserTool.get_next_question(questions)
        if next_question:
            updated_state["current_question"] = next_question.id
    
    # Update conversation history
    if "conversation_history" not in updated_state:
        updated_state["conversation_history"] = []
        
    updated_state["conversation_history"].append({
        "user_message": user_message,
        "intent": context["intent"],
        "timestamp": "now"  # In real implementation, use actual timestamp
    })
    
    return updated_state