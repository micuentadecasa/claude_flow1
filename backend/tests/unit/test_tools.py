import pytest
import os
from pathlib import Path
from agent.tools_and_schemas import (
    DocumentReaderTool,
    QuestionParserTool,
    AnswerEnhancementTool,
    ProgressTrackerTool,
    AuditQuestion
)


class TestDocumentReaderTool:
    """Test document reading and parsing functionality"""
    
    def test_read_document_success(self):
        """Test successful document reading"""
        # Use the actual audit document
        document_path = "cuestionario_auditoria_nes.md"
        result = DocumentReaderTool.read_document(document_path)
        
        assert result["success"] == True
        assert len(result["questions"]) > 0
        assert "document_path" in result
        
        # Check first question structure
        first_question = result["questions"][0]
        assert hasattr(first_question, 'id')
        assert hasattr(first_question, 'content')
        assert hasattr(first_question, 'status')
        assert first_question.status == "pending"
    
    def test_read_document_file_not_found(self):
        """Test handling of missing document"""
        result = DocumentReaderTool.read_document("nonexistent_file.md")
        
        assert result["success"] == False
        assert "error" in result
        assert result["questions"] == []
    
    def test_parse_questions_format(self):
        """Test question parsing format"""
        document_path = "cuestionario_auditoria_nes.md"
        result = DocumentReaderTool.read_document(document_path)
        
        questions = result["questions"]
        for question in questions:
            assert question.content.startswith('¿')
            assert question.content.endswith('?')
            assert question.id.startswith('pregunta_')


class TestQuestionParserTool:
    """Test question status and navigation functionality"""
    
    def setup_method(self):
        """Setup test questions"""
        self.questions = [
            AuditQuestion(id="pregunta_1", title="Pregunta 1", content="¿Test 1?", status="answered"),
            AuditQuestion(id="pregunta_2", title="Pregunta 2", content="¿Test 2?", status="pending"),
            AuditQuestion(id="pregunta_3", title="Pregunta 3", content="¿Test 3?", status="pending"),
        ]
    
    def test_get_question_status(self):
        """Test progress calculation"""
        progress = QuestionParserTool.get_question_status(self.questions)
        
        assert progress.total_questions == 3
        assert progress.answered == 1
        assert progress.pending == 2
        assert progress.completion_percentage == pytest.approx(33.33, rel=1e-2)
    
    def test_get_next_question(self):
        """Test getting next pending question"""
        next_question = QuestionParserTool.get_next_question(self.questions)
        
        assert next_question is not None
        assert next_question.id == "pregunta_2"
        assert next_question.status == "pending"
    
    def test_get_next_question_all_completed(self):
        """Test when all questions are answered"""
        for question in self.questions:
            question.status = "answered"
        
        next_question = QuestionParserTool.get_next_question(self.questions)
        assert next_question is None


class TestAnswerEnhancementTool:
    """Test answer enhancement with NES expertise"""
    
    def test_categorize_backup_question(self):
        """Test categorization of backup-related questions"""
        question = "¿Cómo realiza la empresa las copias de seguridad?"
        answer = "Tenemos un NAS"
        
        enhancement = AnswerEnhancementTool.enhance_answer(question, answer)
        
        assert enhancement["category"] == "backup"
        assert enhancement["enhancement_needed"] == True
        assert len(enhancement["missing_requirements"]) > 0
        assert len(enhancement["suggestions"]) > 0
    
    def test_categorize_access_control_question(self):
        """Test categorization of access control questions"""
        question = "¿Qué mecanismos de control de acceso tiene implementados?"
        answer = "Passwords"
        
        enhancement = AnswerEnhancementTool.enhance_answer(question, answer)
        
        assert enhancement["category"] == "access_control"
        assert enhancement["enhancement_needed"] == True
    
    def test_comprehensive_backup_answer(self):
        """Test comprehensive backup answer"""
        question = "¿Cómo realiza la empresa las copias de seguridad?"
        comprehensive_answer = """
        Utilizamos un sistema QNAP NAS de 8TB con copias diarias automáticas a las 02:00.
        Las copias se verifican semanalmente mediante pruebas de recuperación.
        Mantenemos copias locales por 30 días y copias remotas en AWS S3 por 7 años.
        Tenemos un plan de recuperación documentado con RTO de 4 horas.
        """
        
        enhancement = AnswerEnhancementTool.enhance_answer(question, comprehensive_answer)
        
        # Should need less enhancement
        assert len(enhancement["missing_requirements"]) < 3
    
    def test_generate_suggestions(self):
        """Test suggestion generation"""
        missing_info = ["Tipo de sistema", "Frecuencia de copias"]
        suggestions = AnswerEnhancementTool._generate_suggestions(missing_info)
        
        assert len(suggestions) == 2
        for suggestion in suggestions:
            assert suggestion.startswith("Por favor, proporcione detalles sobre:")


class TestProgressTrackerTool:
    """Test progress tracking and reporting"""
    
    def setup_method(self):
        """Setup test questions"""
        self.questions = [
            AuditQuestion(id="pregunta_1", title="Pregunta 1", content="¿Test 1?", status="answered"),
            AuditQuestion(id="pregunta_2", title="Pregunta 2", content="¿Test 2?", status="answered"),
            AuditQuestion(id="pregunta_3", title="Pregunta 3", content="¿Test 3?", status="pending"),
        ]
    
    def test_get_completion_summary(self):
        """Test completion summary generation"""
        summary = ProgressTrackerTool.get_completion_summary(self.questions)
        
        assert "Estado de la Auditoría" in summary
        assert "Total de preguntas: 3" in summary
        assert "Completadas: 2" in summary
        assert "Pendientes: 1" in summary
        assert "66.7%" in summary
    
    def test_suggest_next_action_with_pending(self):
        """Test next action suggestion with pending questions"""
        suggestion = ProgressTrackerTool.suggest_next_action(self.questions)
        
        assert "Siguiente pregunta recomendada" in suggestion
        assert "Pregunta 3" in suggestion
    
    def test_suggest_next_action_all_completed(self):
        """Test next action when all completed"""
        for question in self.questions:
            question.status = "answered"
        
        suggestion = ProgressTrackerTool.suggest_next_action(self.questions)
        
        assert "Felicitaciones" in suggestion
        assert "completado todas las preguntas" in suggestion


if __name__ == "__main__":
    # Run tests with real LLM calls and file operations
    print("🧪 Running tool tests with real file operations...")
    pytest.main([__file__, "-v"])