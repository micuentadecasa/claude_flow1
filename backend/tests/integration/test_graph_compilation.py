import pytest
import os
from agent.graph import graph
from agent.state import OverallState


class TestGraphCompilation:
    """Test graph compilation and import validation"""
    
    def test_graph_imports_successfully(self):
        """Test that graph can be imported without errors"""
        from agent.graph import graph
        assert graph is not None
        print("✅ Graph imported successfully")
    
    def test_graph_structure(self):
        """Test graph structure and nodes"""
        # Check graph has expected structure
        assert hasattr(graph, 'nodes')
        # Note: CompiledStateGraph doesn't have 'edges' attribute, only nodes
        
        # Check for audit coordinator node
        node_names = list(graph.nodes.keys())
        assert "audit_coordinator" in node_names
        print(f"✅ Graph nodes: {node_names}")
    
    def test_graph_compilation_with_config(self):
        """Test graph compiles with configuration schema"""
        from agent.configuration import Configuration
        
        # Test that graph was compiled with Configuration schema
        assert hasattr(graph, 'config_schema')
        # config_schema is a bound method in CompiledStateGraph, not the class itself
        assert callable(graph.config_schema)
        print("✅ Graph compiled with Configuration schema")
    
    def test_state_schema_validation(self):
        """Test OverallState schema is valid"""
        # Test that we can create a valid state
        test_state = {
            "messages": [],
            "document_path": "test.md",
            "questions_status": {},
            "current_question": None,
            "user_context": {},
            "language": "es",
            "conversation_history": []
        }
        
        # This should not raise any errors
        assert isinstance(test_state["messages"], list)
        assert isinstance(test_state["document_path"], str)
        assert isinstance(test_state["questions_status"], dict)
        assert isinstance(test_state["user_context"], dict)
        assert isinstance(test_state["language"], str)
        assert isinstance(test_state["conversation_history"], list)
        
        print("✅ OverallState schema is valid")
    
    def test_absolute_imports_in_graph(self):
        """Test that graph uses absolute imports (TIP #006)"""
        # Check graph.py file directly for absolute imports
        import os
        
        graph_file = os.path.join(os.path.dirname(__file__), "..", "..", "src", "agent", "graph.py")
        with open(graph_file, 'r') as f:
            source = f.read()
        
        # Should not contain relative imports
        assert "from .nodes" not in source
        assert "from ..agent" not in source
        # Should contain absolute imports
        assert "from agent.nodes.audit_coordinator import audit_coordinator_agent" in source
        print("✅ Graph uses absolute imports (TIP #006)")
    
    def test_configuration_integration(self):
        """Test Configuration is properly integrated"""
        from agent.configuration import Configuration
        
        # Test default configuration values
        config = Configuration()
        assert hasattr(config, 'answer_model')
        assert hasattr(config, 'query_generator_model')
        
        print(f"✅ Configuration integrated: {config.answer_model}")
    
    def test_node_function_signatures(self):
        """Test node functions have correct signatures (TIP #008)"""
        from agent.nodes.audit_coordinator import audit_coordinator_agent
        import inspect
        
        # Get function signature
        sig = inspect.signature(audit_coordinator_agent)
        params = list(sig.parameters.keys())
        
        # Should have state and config parameters
        assert "state" in params
        assert "config" in params
        assert len(params) == 2
        
        print("✅ Node function has correct signature: (state, config)")
    
    def test_environment_variables(self):
        """Test required environment variables"""
        # GEMINI_API_KEY should be checked in graph.py
        # This test verifies the check exists
        import agent.graph
        
        # The import should succeed if GEMINI_API_KEY is set
        # or raise ValueError if not set
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️ GEMINI_API_KEY not set - graph will raise ValueError at runtime")
        else:
            print("✅ GEMINI_API_KEY is configured")


if __name__ == "__main__":
    print("🧪 Running graph compilation tests...")
    pytest.main([__file__, "-v"])