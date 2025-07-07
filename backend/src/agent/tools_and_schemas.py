from typing import List, Dict, Any
from pydantic import BaseModel, Field
import os
import re
from pathlib import Path


class AuditQuestion(BaseModel):
    id: str = Field(description="Unique identifier for the question")
    title: str = Field(description="Question title/section")
    content: str = Field(description="Full question text")
    status: str = Field(description="answered, pending, or needs_improvement")
    answer: str = Field(default="", description="User's answer to the question")


class AuditProgress(BaseModel):
    total_questions: int = Field(description="Total number of questions")
    answered: int = Field(description="Number of answered questions")
    pending: int = Field(description="Number of pending questions")
    completion_percentage: float = Field(description="Completion percentage")


class DocumentReaderTool:
    """Tool for reading and parsing audit questionnaire MD files"""
    
    @staticmethod
    def read_document(file_path: str) -> Dict[str, Any]:
        """Read and parse the audit questionnaire"""
        try:
            # Handle different possible paths
            possible_paths = [
                file_path,
                os.path.join(os.getcwd(), file_path),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), file_path)
            ]
            
            content = None
            actual_path = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        actual_path = path
                        break
            
            if content is None:
                raise FileNotFoundError(f"Could not find file in any of these paths: {possible_paths}")
            
            questions = DocumentReaderTool._parse_questions(content)
            return {
                "success": True,
                "questions": questions,
                "document_path": actual_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "questions": []
            }
    
    @staticmethod
    def _parse_questions(content: str) -> List[AuditQuestion]:
        """Parse questions from markdown content"""
        questions = []
        
        # Split by sections (###)
        sections = re.split(r'\n### Pregunta \d+', content)
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            lines = section.strip().split('\n')
            if lines:
                # Extract question content
                question_content = lines[0].replace('**', '').strip()
                if question_content.startswith('¿'):
                    questions.append(AuditQuestion(
                        id=f"pregunta_{i}",
                        title=f"Pregunta {i}",
                        content=question_content,
                        status="pending"
                    ))
        
        return questions


class QuestionParserTool:
    """Tool for managing question status and navigation"""
    
    @staticmethod
    def get_question_status(questions: List[AuditQuestion]) -> AuditProgress:
        """Get overall progress status"""
        total = len(questions)
        answered = len([q for q in questions if q.status == "answered"])
        pending = len([q for q in questions if q.status == "pending"])
        
        completion = (answered / total * 100) if total > 0 else 0
        
        return AuditProgress(
            total_questions=total,
            answered=answered,
            pending=pending,
            completion_percentage=completion
        )
    
    @staticmethod
    def get_next_question(questions: List[AuditQuestion]) -> AuditQuestion:
        """Get the next pending question"""
        for question in questions:
            if question.status == "pending":
                return question
        return None


class AnswerEnhancementTool:
    """Tool for improving answers using NES expertise"""
    
    NES_EXPERTISE = {
        "backup": {
            "keywords": ["copia", "backup", "respaldo"],
            "requirements": [
                "Tipo de sistema de backup utilizado",
                "Frecuencia de las copias de seguridad",
                "Procedimientos de verificación y pruebas",
                "Ubicación de almacenamiento (local/remoto)",
                "Tiempo de retención de copias",
                "Plan de recuperación ante desastres"
            ]
        },
        "access_control": {
            "keywords": ["acceso", "autenticación", "control"],
            "requirements": [
                "Sistemas de autenticación implementados",
                "Políticas de contraseñas",
                "Autenticación multifactor",
                "Gestión de privilegios y roles",
                "Revisión periódica de accesos",
                "Registro de actividades de acceso"
            ]
        },
        "monitoring": {
            "keywords": ["monitoreo", "detección", "intrusion"],
            "requirements": [
                "Herramientas de monitoreo de red",
                "Sistemas de detección de intrusiones",
                "Análisis de logs y alertas",
                "Procedimientos de respuesta a incidentes",
                "Escalación de alertas",
                "Informes de actividad sospechosa"
            ]
        }
    }
    
    @staticmethod
    def enhance_answer(question: str, answer: str) -> Dict[str, Any]:
        """Enhance user answer with NES expertise"""
        category = AnswerEnhancementTool._categorize_question(question)
        requirements = AnswerEnhancementTool.NES_EXPERTISE.get(category, {}).get("requirements", [])
        
        missing_info = []
        for req in requirements:
            if not AnswerEnhancementTool._check_requirement_covered(answer, req):
                missing_info.append(req)
        
        return {
            "category": category,
            "missing_requirements": missing_info,
            "enhancement_needed": len(missing_info) > 0,
            "suggestions": AnswerEnhancementTool._generate_suggestions(missing_info)
        }
    
    @staticmethod
    def _categorize_question(question: str) -> str:
        """Categorize question based on keywords"""
        question_lower = question.lower()
        for category, data in AnswerEnhancementTool.NES_EXPERTISE.items():
            if any(keyword in question_lower for keyword in data["keywords"]):
                return category
        return "general"
    
    @staticmethod
    def _check_requirement_covered(answer: str, requirement: str) -> bool:
        """Check if answer covers specific requirement"""
        answer_lower = answer.lower()
        req_keywords = requirement.lower().split()
        return any(keyword in answer_lower for keyword in req_keywords)
    
    @staticmethod
    def _generate_suggestions(missing_info: List[str]) -> List[str]:
        """Generate specific suggestions for missing information"""
        suggestions = []
        for info in missing_info:
            suggestions.append(f"Por favor, proporcione detalles sobre: {info}")
        return suggestions


class ProgressTrackerTool:
    """Tool for tracking and reporting audit progress"""
    
    @staticmethod
    def get_completion_summary(questions: List[AuditQuestion]) -> str:
        """Generate completion summary in Spanish"""
        progress = QuestionParserTool.get_question_status(questions)
        
        summary = f"""
📊 **Estado de la Auditoría:**
- Total de preguntas: {progress.total_questions}
- Completadas: {progress.answered}
- Pendientes: {progress.pending}
- Progreso: {progress.completion_percentage:.1f}%
        """.strip()
        
        return summary
    
    @staticmethod
    def suggest_next_action(questions: List[AuditQuestion]) -> str:
        """Suggest next action based on current state"""
        next_question = QuestionParserTool.get_next_question(questions)
        
        if next_question:
            return f"Siguiente pregunta recomendada: {next_question.title}"
        else:
            return "¡Felicitaciones! Has completado todas las preguntas de la auditoría."


class AnswerSaverTool:
    """Tool for saving answers to persistent storage"""
    
    @staticmethod
    def save_answer(question_id: str, answer: str, questions: List[AuditQuestion], file_path: str = "audit_answers.json") -> Dict[str, Any]:
        """Save or update an answer for a specific question"""
        try:
            import json
            from datetime import datetime
            
            # Update the question in the list
            for question in questions:
                if question.id == question_id:
                    question.answer = answer
                    question.status = "answered"
                    break
            else:
                return {"success": False, "error": f"Question {question_id} not found"}
            
            # Load existing answers or create new structure
            answers_data = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        answers_data = json.load(f)
                except:
                    answers_data = {}
            
            # Update answers data
            if "answers" not in answers_data:
                answers_data["answers"] = {}
            
            answers_data["answers"][question_id] = {
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "status": "answered"
            }
            answers_data["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(answers_data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "message": f"Answer saved for {question_id}",
                "file_path": file_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def load_answers(questions: List[AuditQuestion], file_path: str = "audit_answers.json") -> List[AuditQuestion]:
        """Load saved answers and update question status"""
        try:
            if not os.path.exists(file_path):
                return questions
            
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            
            saved_answers = answers_data.get("answers", {})
            
            for question in questions:
                if question.id in saved_answers:
                    answer_info = saved_answers[question.id]
                    question.answer = answer_info.get("answer", "")
                    question.status = answer_info.get("status", "pending")
            
            return questions
            
        except Exception as e:
            print(f"Error loading answers: {e}")
            return questions
