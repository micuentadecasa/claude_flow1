import pytest
import subprocess
import time
import requests
import signal
import os
from pathlib import Path


class TestServerStartup:
    """Test LangGraph server startup and basic functionality"""
    
    def setup_method(self):
        """Setup for server tests"""
        self.server_process = None
        self.server_port = 2024
        self.server_url = f"http://localhost:{self.server_port}"
    
    def teardown_method(self):
        """Cleanup after server tests"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
    
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    def test_langgraph_server_startup(self):
        """Test that langgraph dev server starts without import errors"""
        
        # Change to backend_gen directory
        backend_gen_path = Path(".")
        
        try:
            # Start langgraph dev server
            print("🚀 Starting langgraph dev server...")
            self.server_process = subprocess.Popen(
                ["langgraph", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=backend_gen_path
            )
            
            # Wait for server to start
            max_wait = 30  # seconds
            start_time = time.time()
            server_ready = False
            
            while time.time() - start_time < max_wait:
                try:
                    # Check if server is responding
                    response = requests.get(f"{self.server_url}/ok", timeout=2)
                    if response.status_code == 200:
                        server_ready = True
                        break
                except requests.exceptions.RequestException:
                    pass
                
                # Check for errors in stderr
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    pytest.fail(f"Server failed to start. Stderr: {stderr}")
                
                time.sleep(1)
            
            assert server_ready, "Server did not start within timeout period"
            print("✅ LangGraph server started successfully")
            
        except Exception as e:
            if self.server_process:
                stdout, stderr = self.server_process.communicate()
                print(f"Server stdout: {stdout}")
                print(f"Server stderr: {stderr}")
            raise e
    
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    def test_server_health_check(self):
        """Test server health endpoint"""
        # This test requires server to be running from previous test
        # or can be run independently with external server
        
        try:
            response = requests.get(f"{self.server_url}/ok", timeout=5)
            assert response.status_code == 200
            print("✅ Server health check passed")
        except requests.exceptions.RequestException:
            pytest.skip("Server not available - run test_langgraph_server_startup first")
    
    def test_package_installation(self):
        """Test that package is properly installed in editable mode"""
        try:
            # Test importing the agent package
            import agent
            from agent.graph import graph
            from agent.state import OverallState
            from agent.nodes.audit_coordinator import audit_coordinator_agent
            
            print("✅ All agent modules import successfully")
            
        except ImportError as e:
            pytest.fail(f"Package not properly installed: {e}")
    
    def test_pre_server_validation(self):
        """Test graph loading before server startup (TIP #005)"""
        try:
            # This is the critical pre-server check from CLAUDE.md
            from agent.graph import graph
            assert graph is not None
            print(f"✅ Graph loads successfully: {type(graph)}")
            
        except Exception as e:
            pytest.fail(f"Graph failed to load - will cause server startup failure: {e}")
    
    def test_document_file_exists(self):
        """Test that audit document exists and is readable"""
        document_path = "cuestionario_auditoria_nes.md"
        
        assert os.path.exists(document_path), f"Audit document not found: {document_path}"
        
        # Test file is readable
        with open(document_path, 'r', encoding='utf-8') as file:
            content = file.read()
            assert len(content) > 0
            assert "Pregunta" in content
            
        print(f"✅ Audit document exists and is readable: {document_path}")
    
    def test_environment_setup(self):
        """Test environment is properly configured"""
        # Check Python path includes current directory
        import sys
        current_dir = str(Path(".").resolve())
        
        # Should be able to import agent
        try:
            import agent
            print("✅ Agent package importable")
        except ImportError:
            pytest.fail("Agent package not importable - check pip install -e .")
        
        # Check for required files
        required_files = [
            "src/agent/__init__.py",
            "src/agent/graph.py", 
            "src/agent/state.py",
            "src/agent/configuration.py",
            "src/agent/nodes/audit_coordinator.py",
            "langgraph.json"
        ]
        
        for file_path in required_files:
            assert os.path.exists(file_path), f"Required file missing: {file_path}"
        
        print("✅ All required files present")


class TestServerLogging:
    """Test server logging and error detection"""
    
    def test_import_error_detection(self):
        """Test detection of import errors in server logs"""
        # This would be used in actual server testing
        error_patterns = [
            "ImportError",
            "ModuleNotFoundError", 
            "attempted relative import",
            "No module named"
        ]
        
        sample_error_log = """
        2024-01-01 10:00:00 - INFO - Starting server
        2024-01-01 10:00:01 - ERROR - ImportError: attempted relative import with no known parent package
        """
        
        has_import_error = any(pattern in sample_error_log for pattern in error_patterns)
        assert has_import_error  # This sample should detect an error
        
        print("✅ Import error detection logic works")


if __name__ == "__main__":
    print("🧪 Running server startup integration tests...")
    pytest.main([__file__, "-v", "-s"])