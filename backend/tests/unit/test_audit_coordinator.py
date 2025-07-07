import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_core.runnables import RunnableConfig

from agent.state import OverallState
from agent.nodes.audit_coordinator import audit_coordinator_agent
from agent.configuration import Configuration


class TestAuditCoordinatorAgent:
    """Test audit coordinator agent functionality with real LLM calls"""
    
    def setup_method(self):
        """Setup test state and configuration"""
        # Use Configuration defaults instead of hardcoded values
        default_config = Configuration()
        self.config = RunnableConfig(
            configurable={
                "answer_model": default_config.answer_model,
                "query_generator_model": default_config.query_generator_model,
                "reflection_model": default_config.reflection_model,
                "number_of_initial_queries": default_config.number_of_initial_queries,
                "max_research_loops": default_config.max_research_loops
            }
        )
        
        self.initial_state = {
            "messages": [{"role": "user", "content": "Hola, quiero empezar la auditoría"}],
            "document_path": "cuestionario_auditoria_nes.md",
            "questions_status": {},
            "current_question": None,
            "user_context": {},
            "language": "es",
            "conversation_history": []
        }
    
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    def test_welcome_interaction_real_llm(self):
        """Test welcome interaction with real LLM"""
        result = audit_coordinator_agent(self.initial_state, self.config)
        
        # Check response structure
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check Spanish response
        assistant_message = result["messages"][-1]
        assert assistant_message["role"] == "assistant"
        response_content = assistant_message["content"].lower()
        
        # Should contain Spanish greeting elements
        assert any(word in response_content for word in ["hola", "auditoría", "seguridad", "nes"])
        
        # Should mention total questions or offer to start
        assert any(word in response_content for word in ["preguntas", "empezar", "comenzar"])
        
        print(f"✅ Welcome response: {assistant_message['content'][:100]}...")
    
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    def test_progress_request_real_llm(self):
        """Test progress request with real LLM"""
        state = self.initial_state.copy()
        state["messages"] = [{"role": "user", "content": "¿Cuántas preguntas me quedan?"}]
        state["questions_status"] = {"pregunta_1": "answered", "pregunta_2": "answered"}
        
        result = audit_coordinator_agent(state, self.config)
        
        assistant_message = result["messages"][-1]
        response_content = assistant_message["content"].lower()
        
        # Should contain progress information
        assert any(word in response_content for word in ["preguntas", "completadas", "pendientes"])
        
        print(f"✅ Progress response: {assistant_message['content'][:100]}...")
    
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    def test_answer_enhancement_real_llm(self):
        """Test answer enhancement with real LLM"""
        state = self.initial_state.copy()
        state["messages"] = [{"role": "user", "content": "Tenemos un NAS para backups"}]
        state["current_question"] = "pregunta_1"
        
        result = audit_coordinator_agent(state, self.config)
        
        assistant_message = result["messages"][-1]
        response_content = assistant_message["content"].lower()
        
        # Should ask for more details about NAS setup
        assert any(word in response_content for word in ["detalles", "más", "específico", "frecuencia"])
        
        print(f"✅ Enhancement response: {assistant_message['content'][:100]}...")
    
    def test_state_initialization(self):
        """Test proper state initialization"""
        empty_state = {"messages": []}
        
        # Mock LLM to avoid API call for initialization test
        with patch('agent.nodes.audit_coordinator.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Initialized response"
            mock_llm_class.return_value = mock_llm
            
            result = audit_coordinator_agent(empty_state, self.config)
            
            # Check state initialization
            assert result["document_path"] == "cuestionario_auditoria_nes.md"
            assert result["language"] == "es"
            assert isinstance(result["user_context"], dict)
            assert isinstance(result["conversation_history"], list)
            assert isinstance(result["questions_status"], dict)
    
    def test_document_loading_error_handling(self):
        """Test handling of document loading errors"""
        # Mock the DocumentReaderTool to simulate an error
        with patch('agent.nodes.audit_coordinator.get_audit_status') as mock_get_status:
            mock_get_status.return_value = ("Error loading questionnaire: [Errno 2] No such file or directory", [])
            
            result = audit_coordinator_agent(self.initial_state, self.config)
            
            assistant_message = result["messages"][-1]
            response_content = assistant_message["content"]
            
            # Should contain error message in Spanish or handle gracefully
            assert "cuestionario" in response_content.lower() or "problema" in response_content.lower() or "error" in response_content.lower()
    
    def test_conversation_history_update(self):
        """Test conversation history tracking"""
        with patch('agent.nodes.audit_coordinator.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Test response"
            mock_llm_class.return_value = mock_llm
            
            result = audit_coordinator_agent(self.initial_state, self.config)
            
            # Check conversation history was updated
            assert len(result["conversation_history"]) > 0
            last_entry = result["conversation_history"][-1]
            assert "user" in last_entry
            assert "assistant" in last_entry
            assert "timestamp" in last_entry
    
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set") 
    def test_next_question_flow_real_llm(self):
        """Test next question flow with real LLM"""
        state = self.initial_state.copy()
        state["messages"] = [{"role": "user", "content": "siguiente pregunta"}]
        
        result = audit_coordinator_agent(state, self.config)
        
        assistant_message = result["messages"][-1]
        response_content = assistant_message["content"]
        
        # Should present a question
        assert "¿" in response_content and "?" in response_content
        
        print(f"✅ Next question response: {assistant_message['content'][:100]}...")


if __name__ == "__main__":
    # Run tests with real LLM calls
    print("🧪 Running audit coordinator tests with real LLM calls...")
    pytest.main([__file__, "-v", "-s"])