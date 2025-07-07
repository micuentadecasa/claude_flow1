# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fullstack agentic application generator built on LangGraph, React, and Google Gemini. The system creates intelligent agents that can coordinate complex business workflows. The project includes both a working example (backend/) and a generation system (backend_gen/) for creating new agent architectures.

## Key Architecture

### Dual Backend Structure
- `backend/` - Working LangGraph research agent (web search + reflection) used as base, never use it for testing
- `backend_gen/` - Generated agent workspace for new business cases, use this for testing
- `frontend/` - React/Vite interface for both backends

### Agent Generation System

1. use @docs/roadmap.md for current status and next steps
2. Generates complete LangGraph applications in `backend_gen/`
3. Follows structured phases with comprehensive testing
4. Accumulates knowledge in `/docs/tips.md`

## Development Commands

### Main Development
```bash
# Start both frontend and working backend (research agent)
make dev

# Start frontend and generated backend (use case implementation)
make gen

# Frontend only
make dev-frontend

# Working backend only  
make dev-backend

# Generated backend only
make dev-backend-gen
```

**CRITICAL**: For generated agent development, ALWAYS use `make gen` not `make dev`. The `make gen` command works with the generated backend in `backend_gen/` folder.

### Backend Development (Python/LangGraph)
```bash
cd backend  # or backend_gen
pip install -e .           # Install in editable mode
langgraph dev             # Start LangGraph dev server (port 2024)
python -m pytest tests/  # Run comprehensive tests
ruff check src/          # Lint code
```

### Frontend Development (React/TypeScript)
```bash
cd frontend
npm install              # Install dependencies
npm run dev             # Start Vite dev server (port 5173)
npm run build           # Build for production
npm run lint            # ESLint code
```

## Important File Patterns

### LangGraph Agent Structure
Required files in any backend implementation:
- `src/agent/state.py` - OverallState TypedDict definition
- `src/agent/graph.py` - Graph assembly with absolute imports
- `src/agent/nodes/` - Individual agent node functions
- `src/agent/tools_and_schemas.py` - Pydantic models and tools
- `src/agent/configuration.py` - LLM model configuration
- `langgraph.json` - Deployment configuration

### Critical Development Patterns

#### LLM Configuration Pattern
Always use configuration-based model selection:
```python
from agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig

def node_function(state: OverallState, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.answer_model,  # Use configured model
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY")
    )
```

**Configuration Fields Available**:
- `query_generator_model`: Default "gemini-2.0-flash"
- `reflection_model`: Default "gemini-2.5-flash-preview-04-17" 
- `answer_model`: Default "gemini-1.5-flash-latest"
- `number_of_initial_queries`: Default 3
- `max_research_loops`: Default 2

#### Import Requirements
- Use ABSOLUTE imports in graph.py: `from agent.nodes.x import y`
- Never use relative imports: `from .nodes.x import y` (breaks langgraph dev)
- Keep agent/__init__.py minimal to prevent circular imports

#### State Management
All agents share OverallState TypedDict with consistent field patterns:
```python
from langgraph.graph import add_messages
from typing_extensions import Annotated

class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    document_path: str
    questions_status: Dict[str, str]  # question_id -> "answered"|"pending"|"needs_improvement"
    current_question: Optional[str]
    user_context: Dict[str, Any]
    language: str
    conversation_history: List[Dict[str, Any]]
```

## Testing Strategy

### Comprehensive Testing Phases
1. **Unit Testing** - Individual agent functions with real LLM calls
2. **Graph Compilation** - Import validation and graph building
3. **Server Testing** - LangGraph dev server with real execution
4. **API Integration** - REST endpoint validation
5. **Scenario Testing** - Comprehensive business case scenarios with domain-specific validation

### Critical Testing Commands
```bash
# Pre-server validation (prevents runtime failures)
python -c "from agent.graph import graph; print('Graph loads successfully')"
pip install -e .

# Server testing with proper cleanup
langgraph dev > langgraph.log 2>&1 &
SERVER_PID=$!
# ... testing logic ...
kill $SERVER_PID
```

**CRITICAL SERVER VALIDATION**: After ANY changes to agent code, ALWAYS check console output:
```bash
# 1. Start server in background and monitor logs
nohup langgraph dev > langgraph.log 2>&1 & 
sleep 5

# 2. Check for import/runtime errors in console
tail -10 langgraph.log | grep -i "error\|exception\|failed"

# 3. Test actual endpoint execution
curl -X POST "http://127.0.0.1:2024/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "agent", "input": {"messages": [{"role": "user", "content": "test"}]}, "stream_mode": "values"}'

# 4. Verify no errors in response (should not contain "error" event)
```

The server may start successfully but still have runtime errors when graph execution begins. Always test actual execution, not just server startup.

### LangWatch Scenario Testing & Agent Simulation

**CRITICAL FRAMEWORK**: Use LangWatch Scenario library for sophisticated agent testing through realistic user simulation.

**What is LangWatch Scenario**: Advanced Agent Testing Framework based on simulations that can:
- Test real agent behavior by simulating users in different scenarios and edge cases  
- Evaluate and judge at any point of the conversation with powerful multi-turn control
- Combine with any LLM eval framework or custom evals (agnostic by design)
- Integrate any agent by implementing just one `call()` method
- Available in Python, TypeScript and Go with comprehensive testing capabilities

#### **Installation & Setup**
```bash
# Install LangWatch Scenario framework
cd /backend_gen
pip install langwatch-scenario pytest

# Verify installation
python -c "import scenario; print('LangWatch Scenario installed successfully')"

# Set up environment variables for LangWatch
echo "LANGWATCH_API_KEY=your-api-key-here" >> .env
echo "OPENAI_API_KEY=your-openai-key-here" >> .env  # Required for user simulation
# Note: GEMINI_API_KEY in .env is used for our agent, OPENAI_API_KEY is for LangWatch user simulation

# Configure scenario defaults
python -c "
import scenario
scenario.configure(
    default_model='openai/gpt-4o-mini',  # For user simulation (most compatible)
    cache_key='spanish-audit-coordination-tests',  # For repeatable tests
    verbose=True  # Show detailed simulation output
)
print('LangWatch Scenario configured')
"
```

**IMPORTANT**: LangWatch Scenario framework works best with OpenAI models for user simulation. While our Spanish audit agent uses Google Gemini (via GEMINI_API_KEY), the user simulation requires OPENAI_API_KEY. If only GEMINI_API_KEY is available, the framework falls back to direct agent testing without user simulation.

#### **Agent Adapter Implementation**
```python
# Create LangWatch Scenario adapter for our Spanish Audit agent
import scenario
import asyncio
from typing import Dict, Any
from agent.nodes.audit_coordinator import audit_coordinator_agent
from agent.configuration import Configuration

# Configure scenario for Spanish audit testing
scenario.configure(
    default_model="openai/gpt-4o-mini",
    cache_key="spanish-audit-nes-v1",
    verbose=True
)

class SpanishAuditCoordinatorAgent(scenario.AgentAdapter):
    """LangWatch Scenario adapter for our Spanish NES Audit LangGraph agent"""
    
    def __init__(self):
        default_config = Configuration()
        self.config = RunnableConfig(
            configurable={
                "answer_model": default_config.answer_model,
                "reflection_model": default_config.reflection_model,
            }
        )
        
    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        """
        Adapter method that LangWatch Scenario calls to interact with our agent.
        Converts scenario input to our Spanish audit agent format.
        """
        # Convert scenario messages to our state format
        state = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in input.messages],
            "document_path": "cuestionario_auditoria_nes.md",
            "questions_status": {},
            "current_question": None,
            "user_context": {},
            "language": "es",
            "conversation_history": []
        }
        
        try:
            # Execute our Spanish audit coordinator agent
            result = audit_coordinator_agent(state, self.config)
            
            # Extract the final response message
            if result.get("messages") and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                if isinstance(final_message, dict) and "content" in final_message:
                    return final_message["content"]
                else:
                    return str(final_message)
                    
            return "Auditoría NES completada."
            
        except Exception as e:
            return f"Error en la auditoría de seguridad: {str(e)}"
```

#### **Spanish Audit Scenario Tests**
```python
# Test Scenarios for Spanish NES Security Audit System

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_routine_backup_audit_scenario():
    """Test audit for standard backup procedures scenario"""
    
    result = await scenario.run(
        name="routine_backup_audit",
        description="""
            Una empresa mediana necesita completar una auditoría de seguridad NES.
            El usuario responde sobre sus procedimientos de copias de seguridad.
            El sistema debe evaluar si cumplen con los estándares NES españoles
            y solicitar detalles específicos cuando la información sea incompleta.
        """,
        agents=[
            SpanishAuditCoordinatorAgent(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent should communicate entirely in Spanish",
                    "Agent should demonstrate knowledge of NES security standards",
                    "Agent should identify incomplete backup information",
                    "Agent should request specific details: frequency, verification, remote storage",
                    "Agent should NOT accept vague answers like 'tenemos un NAS'",
                    "Agent should maintain professional security consultant tone"
                ]
            ),
        ],
        max_turns=8,
        set_id="spanish-audit-nes-tests",
    )
    
    assert result.success, f"Routine backup audit failed: {result.failure_reason}"

@pytest.mark.agent_test  
@pytest.mark.asyncio
async def test_incomplete_access_control_scenario():
    """Test access control audit with incomplete user responses"""
    
    result = await scenario.run(
        name="incomplete_access_control_audit",
        description="""
            Un usuario proporciona información incompleta sobre controles de acceso.
            Dice solo 'tenemos contraseñas para cada empleado'. El sistema debe
            identificar que falta información crítica según NES: MFA, políticas,
            auditorías, gestión de privilegios, etc.
        """,
        agents=[
            SpanishAuditCoordinatorAgent(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent should identify missing NES access control requirements",
                    "Agent should ask about MFA (autenticación multifactor)",
                    "Agent should inquire about privilege management (gestión de privilegios)",
                    "Agent should request information about audit logs (registros de auditoría)",
                    "Agent should ask about password policies (políticas de contraseñas)",
                    "Agent should maintain conversational Spanish throughout"
                ]
            ),
        ],
        max_turns=6,
        script=[
            scenario.user("¿Qué necesitas saber sobre control de acceso?"),
            scenario.agent(),  # Agent asks the access control question
            scenario.user("Tenemos contraseñas y usuarios diferentes para cada empleado"),
            scenario.agent(),  # Agent should identify incomplete answer
            scenario.judge(),  # Evaluate if agent properly identified missing NES requirements
        ],
        set_id="spanish-audit-nes-tests",
    )
    
    assert result.success, f"Access control audit failed: {result.failure_reason}"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_comprehensive_security_audit_flow():
    """Test complete audit flow from start to finish"""
    
    result = await scenario.run(
        name="comprehensive_security_audit", 
        description="""
            Flujo completo de auditoría NES desde el inicio hasta varias preguntas.
            El usuario debe navegar por múltiples secciones: copias de seguridad,
            control de acceso, monitoreo. El sistema debe mantener contexto y
            progreso a través de toda la conversación.
        """,
        agents=[
            SpanishAuditCoordinatorAgent(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent should start audit professionally in Spanish",
                    "Agent should present questions in logical NES order",
                    "Agent should track progress through multiple questions", 
                    "Agent should transition between sections smoothly",
                    "Agent should provide helpful guidance when user asks for help",
                    "Agent should maintain audit context across entire conversation"
                ]
            ),
        ],
        max_turns=15,  # Extended for complete audit flow
        set_id="spanish-audit-nes-tests",
    )
    
    assert result.success, f"Comprehensive audit flow failed: {result.failure_reason}"

# Advanced Scenario with Custom NES Validation
def check_nes_compliance_knowledge(state: scenario.ScenarioState):
    """Custom assertion to check if NES security knowledge was demonstrated"""
    conversation = " ".join([msg.content for msg in state.messages if hasattr(msg, 'content')])
    
    # Check for key NES security indicators
    nes_knowledge_checks = [
        "nes" in conversation.lower() or "esquema nacional" in conversation.lower(),
        "frecuencia" in conversation.lower() and "verificación" in conversation.lower(),
        "mfa" in conversation.lower() or "multifactor" in conversation.lower(),
        "auditoría" in conversation.lower() or "logs" in conversation.lower()
    ]
    
    assert any(nes_knowledge_checks), "Agent did not demonstrate adequate NES security knowledge"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_nes_expertise_validation():
    """Test that NES security expertise is properly demonstrated"""
    
    result = await scenario.run(
        name="nes_expertise_validation",
        description="""
            Validar que el agente demuestra conocimiento experto en estándares NES.
            Debe identificar requisitos específicos y usar terminología técnica apropiada.
        """,
        agents=[
            SpanishAuditCoordinatorAgent(),
            scenario.UserSimulatorAgent(),
        ],
        script=[
            scenario.user("Quiero empezar la auditoría de seguridad"),
            scenario.agent(),  # Agent responds with NES expertise
            scenario.user("¿Qué necesitas saber sobre nuestras copias de seguridad?"),   
            scenario.agent(),  # Agent demonstrates NES backup requirements knowledge
            check_nes_compliance_knowledge,  # Custom NES knowledge check
            scenario.succeed(),  # End successfully if NES knowledge demonstrated
        ],
        set_id="spanish-audit-nes-tests",
    )
    
    assert result.success, f"NES expertise validation failed: {result.failure_reason}"
```

#### **Execution Commands**
```bash
# Run all LangWatch Scenario tests (requires OPENAI_API_KEY for user simulation)
cd /backend_gen
python -m pytest tests/scenarios/test_audit_flow_scenarios.py -v -s --tb=short

# Run basic audit scenarios (works with GEMINI_API_KEY only)
python -m pytest tests/scenarios/test_audit_flow_scenarios.py::TestBasicAuditScenarios -v -s

# Run specific direct test that works without user simulation
python -m pytest tests/scenarios/test_audit_flow_scenarios.py::TestBasicAuditScenarios::test_backup_question_direct -v -s

# Run with OpenAI key for full user simulation scenarios
OPENAI_API_KEY=your-key python -m pytest tests/scenarios/test_audit_flow_scenarios.py::TestSpanishAuditFlowScenarios -v -s
```

**Testing Levels Available**:
1. **Basic Agent Testing**: Uses GEMINI_API_KEY, tests agent directly without user simulation
2. **Full Scenario Testing**: Requires OPENAI_API_KEY, includes realistic user simulation with AI judges
3. **Direct Integration Testing**: Fallback mode when LangWatch scenarios fail, still validates core functionality

## Environment Setup

### Required Environment Variables
```bash
# Backend (.env file)
GEMINI_API_KEY=your_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here  # Optional for tracing
OPENAI_API_KEY=your_openai_key_here        # Required for LangWatch user simulation
```

### API Key Configuration
The system requires Google Gemini API key. LLM libraries do NOT automatically load from .env, so explicitly pass:
```python
api_key=os.getenv("GEMINI_API_KEY")
```

## Business Case Generation Protocol

The `/docs/planning.md` contains a comprehensive protocol for autonomous agent generation:

1. **Business Case Creation** - Generate diverse business scenarios
2. **Architecture Planning** - Design agent networks and workflows  
3. **Implementation** - Generate complete LangGraph applications
4. **Testing & Validation** - Comprehensive multi-phase testing
5. **Knowledge Accumulation** - Document patterns in `/docs/tips.md`

### Key Execution Principle
The system is designed for FULL AUTONOMY - no user confirmation required once started. Each business case follows structured phases with clear success criteria.

## 🔄 MAJOR ARCHITECTURE SHIFT: Modern LLM-First Agent Design

### **⚡ BREAKING CHANGE: From Script-Based to Conversation-Native Design**

**CRITICAL PARADIGM SHIFT**: We discovered that traditional agent architectures over-engineer what modern LLMs can handle naturally. This represents a fundamental change in how we build conversational agents.

#### **📊 Impact Summary:**
- **Code Reduction**: 70% less code (350+ lines → 135 lines)
- **Better UX**: More natural, flexible conversations
- **Faster Development**: Focus on domain expertise, not conversation engineering
- **Easier Maintenance**: Single prompt updates vs. multiple helper functions

### **Conversational-First Approach**

**CRITICAL PRINCIPLE**: Modern LLMs are capable of natural conversation understanding. Avoid over-engineering with scripted conversation flows, intent detection, or complex routing logic.

#### **✅ DO: Let LLMs Handle Conversation Naturally**

```python
# ✅ PREFERRED: Single comprehensive prompt with embedded expertise
def audit_coordinator_agent(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    # Get configuration and LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.reflection_model,
        temperature=0.1,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Get current audit status using tools
    questions_status, questions_list = get_audit_status()
    
    # Extract user message with proper LangChain message handling
    messages = state.get("messages", [])
    latest_user_message = "Hola"
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == "human":
            latest_user_message = msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_message = msg.get("content", "Hola")
    
    # Single comprehensive prompt with embedded NES expertise
    prompt = f"""Eres un asistente experto en auditorías de seguridad según el estándar NES (Esquema Nacional de Seguridad) de España.

ESTADO ACTUAL DEL CUESTIONARIO:
{questions_status}

CONOCIMIENTO NES PARA EVALUAR RESPUESTAS:
- Copias de seguridad: Requiere tipo de sistema, frecuencia, verificación, ubicación (local/remota), retención, plan de recuperación
- Control de acceso: Autenticación implementada, políticas de contraseñas, MFA, gestión de privilegios, revisiones periódicas, logs
- Monitoreo: Herramientas de red, sistemas de detección, análisis de logs, procedimientos de respuesta, escalación, informes

INSTRUCCIONES:
1. SIEMPRE responde en español, tono conversacional y profesional
2. Si es primer saludo: Da bienvenida y muestra primera pregunta pendiente  
3. Si usuario responde a pregunta: Evalúa completitud contra requisitos NES arriba
4. Si respuesta completa: Di que la guardarás y muestra siguiente pregunta
5. Si respuesta incompleta: Pide específicamente qué falta según NES

MENSAJE DEL USUARIO: {latest_user_message}

Analiza el mensaje y responde apropiadamente."""

    response = llm.invoke(prompt)
    return {"messages": state.get("messages", []) + [{"role": "assistant", "content": response.content}]}
```

#### **❌ AVOID: Over-Engineered Conversation Management**

```python
# ❌ WRONG: Complex intent detection and routing
def old_style_agent(state, config):
    intent = analyze_user_intent(user_message)  # Unnecessary
    
    if intent == "greeting":
        return generate_greeting_response()
    elif intent == "question": 
        return generate_question_response()
    elif intent == "answer":
        return generate_answer_enhancement()
    # ... complex routing logic
```

### **Tool Usage Guidelines**

#### **Use Tools for Actual Operations, Not Logic**

Tools should perform concrete actions, not replace LLM reasoning:

```python
# ✅ CORRECT: Tools for actual operations from backend_gen
class DocumentReaderTool:
    """Tool for reading and parsing audit questionnaire MD files"""
    
    @staticmethod
    def read_document(file_path: str) -> Dict[str, Any]:
        """Read and parse the audit questionnaire"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            questions = DocumentReaderTool._parse_questions(content)
            return {"success": True, "questions": questions, "document_path": file_path}
        except Exception as e:
            return {"success": False, "error": str(e), "questions": []}

class AnswerSaverTool:
    """Tool for saving and loading audit answers"""
    
    @staticmethod
    def save_answer(question_id: str, answer: str, questions: List[AuditQuestion]) -> Dict[str, Any]:
        """Save answer to JSON file"""
        try:
            # Load existing answers
            answers_data = AnswerSaverTool._load_answers_file()
            answers_data[question_id] = answer
            
            # Save to file
            with open("audit_answers.json", "w", encoding="utf-8") as f:
                json.dump(answers_data, f, ensure_ascii=False, indent=2)
            
            return {"success": True, "message": f"Answer saved for {question_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

```python
# ❌ WRONG: Tools for simple logic/formatting
@tool
def analyze_user_intent(message: str) -> str:
    """This should be handled by LLM naturally"""
    if "hello" in message.lower():
        return "greeting"
    # LLM can do this better naturally
```

## 🧪 COMPREHENSIVE TESTING REQUIREMENTS FOR LANGGRAPH AGENTS

### **Critical Testing Checklist for Any LangGraph Solution**

Based on lessons learned from the Spanish NES audit agent, every new LangGraph solution MUST implement these testing patterns to ensure reliability and prevent common failures.

#### **📋 1. Test Structure Requirements**

**MANDATORY Test Suite Structure:**
```
tests/
├── unit/                    # Individual component tests
│   ├── test_agent_nodes.py
│   └── test_tools.py
├── integration/             # Graph and server tests
│   ├── test_graph_compilation.py
│   └── test_server_startup.py
├── conversation_memory/     # Critical memory pattern tests
│   └── test_memory_patterns.py
└── scenarios/               # LangWatch scenario tests
    └── test_business_scenarios.py
```

#### **📦 2. Required Dependencies**

**Add to pyproject.toml:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
asyncio_mode = "auto"

[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "langwatch-scenario>=0.7.0",  # For advanced scenario testing
]
```

#### **🔧 3. Agent State Testing Patterns**

**CRITICAL: Always Return Complete State**
```python
def your_agent_node(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    # ... agent logic ...
    
    # ✅ MUST return ALL required state fields, not just messages
    return {
        "messages": updated_messages,
        "document_path": state.get("document_path", "default.md"),
        "questions_status": updated_status,
        "current_question": next_question_id,
        "user_context": state.get("user_context", {}),
        "language": state.get("language", "en"),
        "conversation_history": updated_history
    }
```

**Unit Test Pattern:**
```python
class TestAgentNode:
    def setup_method(self):
        """Always use Configuration defaults"""
        default_config = Configuration()
        self.config = RunnableConfig(
            configurable={
                "answer_model": default_config.answer_model,
                "reflection_model": default_config.reflection_model,
            }
        )
    
    def test_state_initialization(self):
        """Test that agent returns all required state fields"""
        empty_state = {"messages": []}
        
        with patch('your_module.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "Test response"
            mock_llm_class.return_value = mock_llm
            
            result = your_agent_node(empty_state, self.config)
            
            # Verify ALL expected state fields are present
            assert "messages" in result
            assert "document_path" in result
            assert "conversation_history" in result
            assert isinstance(result["conversation_history"], list)
```

## 🚨 CRITICAL LESSONS LEARNED: Conversation Memory & Context Management

### **🎯 MAJOR DISCOVERY: Multi-Turn Conversation Memory Failures**

During production testing of the Spanish NES audit assistant, we discovered critical conversation memory issues that affect all conversational agents. These patterns must be tested and prevented in every LangGraph application.

#### **Problem #1: Last Message Only Processing**

**Issue**: Agent only processes the latest user message instead of full conversation context.

**✅ SOLUTION: Full Conversation Context Processing**
```python
def agent_node(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    # ❌ WRONG - Only latest message
    latest_user_message = state.get("messages", [])[-1].content
    
    # ✅ CORRECT - Full conversation context
    messages = state.get("messages", [])
    conversation_context = []
    latest_user_message = "Hello"
    
    for msg in messages:
        if hasattr(msg, 'type'):
            if msg.type == "human":
                conversation_context.append(f"Usuario: {msg.content}")
                latest_user_message = msg.content
            elif msg.type == "ai":
                conversation_context.append(f"Asistente: {msg.content}")
    
    conversation_summary = "\n".join(conversation_context[-10:])
    
    prompt = f"""CONVERSATION HISTORY:
{conversation_summary}

INSTRUCTIONS:
- REVIEW conversation history for complete context
- DO NOT repeat questions already answered
- ACCUMULATE information provided across multiple messages
- Evaluate completeness considering ALL information provided

LATEST MESSAGE: {latest_user_message}"""
```

### **🧪 MANDATORY CONVERSATION MEMORY TESTS**

**CRITICAL**: Based on user feedback showing agents still fail conversation memory despite passing tests, every LangGraph application MUST include these test scenarios:

#### **Test #1: Multi-Turn Information Accumulation**
```python
@pytest.mark.conversation_memory
def test_multi_turn_information_accumulation():
    """Test agent accumulates information across multiple messages"""
    
    messages = [
        {"role": "user", "content": "We have a plan"},
        {"role": "assistant", "content": "Tell me more about the plan details"},
        {"role": "user", "content": "RTO is 2 hours"},
        {"role": "assistant", "content": "Good, what about testing frequency?"},
        {"role": "user", "content": "We test annually"}
    ]
    
    state = {"messages": messages, "current_topic": "business_continuity"}
    result = agent_function(state, config)
    
    # Agent should recognize complete answer from multiple turns
    assert "complete" in result.get("status", "").lower()
    assert "2 hours" in result.get("accumulated_answer", "")
    assert "annually" in result.get("accumulated_answer", "")
```

#### **Test #2: No Repetitive Questions**
```python
@pytest.mark.conversation_memory  
def test_no_repetitive_questions():
    """Test agent doesn't ask for already provided information"""
    
    messages = [
        {"role": "user", "content": "We use Entra ID for authentication"},
        {"role": "assistant", "content": "Great! Any MFA requirements?"},
        {"role": "user", "content": "Yes, all critical systems require MFA"}
    ]
    
    state = {"messages": messages}
    result = agent_function(state, config)
    
    # Agent should NOT ask about authentication again
    response = result["messages"][-1]["content"].lower()
    assert "authentication" not in response
    assert "entra id" not in response
    # Should ask about remaining requirements
    assert any(keyword in response for keyword in ["policies", "privileges", "logs"])
```

#### **Test #5: Real-World Conversation Flow Simulation**
```python
@pytest.mark.conversation_memory
def test_real_world_conversation_flow():
    """Test agent with actual reported failure pattern from user feedback"""
    
    # Simulate exact user-reported conversation that failed
    messages = [
        {"role": "assistant", "content": "¿Qué necesitas saber sobre control de acceso?"},
        {"role": "user", "content": "Tenemos usuarios y contraseñas"},
        {"role": "assistant", "content": "¿Podrías contarme más detalles sobre la autenticación?"},
        {"role": "user", "content": "Usamos Entra ID para autenticación"},
        {"role": "assistant", "content": "¿Tenéis autenticación multifactor (MFA) configurada?"},
        {"role": "user", "content": "Sí, MFA está activado para todos los sistemas críticos"},
        {"role": "assistant", "content": "¿Qué políticas de gestión de privilegios tenéis?"},
        {"role": "user", "content": "Los privilegios se revisan cada trimestre"},
        # Now user asks about content - this should NOT repeat previous questions
        {"role": "user", "content": "¿Puedes decirme qué contenidos incluye normalmente un curso de ciberseguridad?"}
    ]
    
    state = {"messages": messages}
    result = agent_function(state, config)
    
    # CRITICAL: Agent should NOT ask about authentication details again
    response = result["messages"][-1]["content"].lower()
    assert "entra id" not in response, "Agent should not ask about Entra ID again"
    assert "mfa" not in response, "Agent should not ask about MFA again"
    assert "autenticación" not in response, "Agent should not ask about authentication again"
    
    # Should respond appropriately to the training content question
    assert any(keyword in response for keyword in ["contenido", "curso", "formación"]), \
        "Agent should address the training content question"
```

#### **Test #6: Complete Conversation Memory Integration Test**
```python
@pytest.mark.conversation_memory
def test_complete_conversation_memory_integration():
    """Comprehensive test covering all conversation memory patterns"""
    
    # Simulate a complete audit conversation with multiple topics
    messages = [
        # Start with backups
        {"role": "assistant", "content": "¡Hola! Empecemos con la auditoría. ¿Cómo gestionáis las copias de seguridad?"},
        {"role": "user", "content": "Tenemos un QNAP"},
        {"role": "assistant", "content": "¿Podrías darme más detalles sobre el sistema QNAP?"},
        {"role": "user", "content": "Es de 8TB, hacemos copias diarias"},
        {"role": "assistant", "content": "¿Verificáis las copias y tenéis plan de recuperación?"},
        {"role": "user", "content": "Sí, verificamos semanalmente y tenemos plan con RTO de 4 horas"},
        {"role": "assistant", "content": "Perfecto. Pasemos al control de acceso. ¿Qué sistema usáis?"},
        
        # Move to access control
        {"role": "user", "content": "Usamos Entra ID"},
        {"role": "assistant", "content": "¿Tenéis MFA y gestión de privilegios?"},
        {"role": "user", "content": "Sí, MFA obligatorio y revisamos privilegios trimestralmente"},
        
        # User asks for help - should not lose context
        {"role": "user", "content": "¿Qué más necesito según NES para control de acceso?"}
    ]
    
    state = {"messages": messages}
    result = agent_function(state, config)
    
    # CRITICAL CHECKS:
    
    # 1. Should remember backup information completely
    backup_info = result.get("topic_data", {}).get("backups", "")
    assert "qnap" in backup_info.lower()
    assert "8tb" in backup_info.lower()
    assert "diarias" in backup_info.lower()
    assert "4 horas" in backup_info.lower()
    
    # 2. Should remember access control information
    access_info = result.get("topic_data", {}).get("access_control", "")
    assert "entra id" in access_info.lower()
    assert "mfa" in access_info.lower()
    assert "trimestral" in access_info.lower()
    
    # 3. Should NOT repeat any previous questions
    response = result["messages"][-1]["content"].lower()
    forbidden_repeats = [
        "qnap", "frecuencia", "verificáis", "entra id", "mfa", "privilegios"
    ]
    for repeat in forbidden_repeats:
        assert repeat not in response, f"Agent should not ask about {repeat} again"
    
    # 4. Should provide NEW NES requirements not yet covered
    nes_requirements = ["logs", "auditoría", "políticas", "revisión", "contraseñas"]
    assert any(req in response for req in nes_requirements), \
        "Agent should suggest additional NES requirements"
```

### **📋 CONVERSATION MEMORY CHECKLIST**

**CRITICAL**: Based on user feedback showing agents still fail conversation memory despite passing tests, every LangGraph application MUST verify:

- [ ] **Full Context Processing**: Agent processes conversation history, not just latest message
- [ ] **Information Accumulation**: Combines partial answers across multiple turns  
- [ ] **No Repetitive Questions**: Doesn't ask for already provided information
- [ ] **Correct Association**: Saves information to appropriate topics/entities
- [ ] **Context Instructions**: Explicit prompt instructions for conversation memory
- [ ] **Topic Identification**: Robust logic to identify current discussion topic
- [ ] **Context Window**: Manages long conversations with appropriate summarization
- [ ] **Memory Tests**: Comprehensive test coverage for conversation scenarios
- [ ] **Real-World Testing**: Tests with exact user conversation patterns that previously failed
- [ ] **Anti-Repetition Enforcement**: Explicit validation that agent doesn't repeat questions

## 📚 LESSONS LEARNED: Critical Agent Design Discoveries

### **🚨 Major Mistakes We Made (And How to Avoid Them)**

#### **❌ MISTAKE #1: Over-Engineering Intent Detection**

**What We Did Wrong:**
```python
# ❌ WRONG: Complex intent analysis
def _analyze_user_intent(user_message, questions, state):
    if any(keyword in user_lower for keyword in ["estado", "progreso"]):
        return {"intent": "status_check", "type": "progress"}
    elif any(keyword in user_lower for keyword in ["siguiente", "próxima"]):
        return {"intent": "next_question", "type": "navigation"}
    # ... 50+ lines of scripted logic
```

**Why This Was Wrong:**
- LLMs naturally understand user intent from context
- Rigid keyword matching misses nuanced user expressions
- Creates unnecessary complexity and maintenance burden

**🎯 LESSON**: Modern LLMs excel at intent understanding. Don't script what they can reason.

#### **❌ MISTAKE #2: Tool Classes for Simple Logic**

**What We Did Wrong:**
```python
# ❌ WRONG: Tool class for domain knowledge
class AnswerEnhancementTool:
    NES_EXPERTISE = {"backup": {"requirements": [...]}}
    
    @staticmethod
    def enhance_answer(question, answer):
        # Just structured data comparison
        return {"missing": [...], "suggestions": [...]}
```

**Why This Was Wrong:**
- Tools should perform actions, not replace LLM reasoning
- Domain knowledge belongs in prompts where LLM can use it flexibly

**🎯 LESSON**: Tools are for operations (save/load), not for domain logic that LLMs can reason about.

### **✅ KEY ARCHITECTURAL DISCOVERIES**

#### **🔍 DISCOVERY #1: LLMs as Natural Conversation Managers**

**Before**: Scripted conversation flows with explicit state tracking
```python
state["current_intent"] = "answer_question"
state["conversation_stage"] = "collecting_details"
state["last_question_type"] = "backup_procedures"
```

**After**: Let LLM track conversation context naturally (backend_gen approach)
```python
# Simple state - LLM manages conversation flow from messages[] and context
class OverallState(TypedDict):
    messages: Annotated[list, add_messages]  # LLM reads conversation history
    document_path: str                       # Current audit questionnaire
    questions_status: Dict[str, str]         # Data state only
    # No conversation flow state needed - LLM handles this
```

**Impact**: Eliminated 60% of state management code.

#### **🔍 DISCOVERY #2: Embedded Expertise > Tool Abstraction**

**Before**: Domain knowledge in separate tool classes
```python
class SecurityExpertise:
    def get_backup_requirements(self): return [...]
    def validate_access_control(self): return [...]
    def assess_monitoring(self): return [...]
```

**After**: Knowledge directly in prompts where LLM can reason (backend_gen approach)
```python
# Embedded in single comprehensive prompt - actual NES expertise
NES_EXPERTISE = """
- Copias de seguridad: Requiere tipo de sistema, frecuencia, verificación, ubicación (local/remota), retención, plan de recuperación
- Control de acceso: Autenticación implementada, políticas de contraseñas, MFA, gestión de privilegios, revisiones periódicas, logs
- Monitoreo: Herramientas de red, sistemas de detección, análisis de logs, procedimientos de respuesta, escalación, informes
"""
# LLM uses this knowledge flexibly based on user input context
```

**Impact**: More flexible application of domain knowledge.

### **📈 MEASURABLE IMPROVEMENTS**

#### **Code Metrics:**
- **Lines of Code**: 350+ → 135 (61% reduction)
- **Functions**: 15+ → 4 (73% reduction)
- **Complexity**: High → Low (single prompt vs. multiple paths)

#### **User Experience:**
- **Conversation Flow**: Rigid → Natural
- **Response Flexibility**: Limited → Adaptive
- **Error Handling**: Scripted → Intelligent

## Critical Development Tips & Error Solutions

### TIP #001: Project Initialization Template
**Category**: Architecture | **Severity**: Critical

**Solution Implementation**:
```bash
# Always start with clean reset
rm -rf tasks
mkdir -p tasks/artifacts
rm -rf backend_gen
cp -r backend backend_gen
cd backend_gen && pip install -e .
```

### TIP #006: Critical LangGraph Server Import Errors and Solutions
**Category**: Development | **Severity**: Critical

**Problem Description**: Multiple specific import errors that only surface when `langgraph dev` server actually runs:
1. **Relative Import Error**: `ImportError: attempted relative import with no known parent package`
2. **Fake Model Import Errors**: Cannot import FakeListChatModel, FakeChatModel
3. **Module Path Errors**: `No module named 'langchain_core.language_models.llm'`

**Solution Implementation**:

Fix Relative Imports in graph.py:
```python
# ❌ WRONG - Relative imports (will fail in server)
from .nodes.audit_coordinator import audit_coordinator_agent

# ✅ CORRECT - Absolute imports (works in server)
from agent.nodes.audit_coordinator import audit_coordinator_agent
```

Fix Fake LLM Imports:
```python
def get_llm():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Try real LLM first
    if os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY")
        )
    
    # Fallback to working fake model
    try:
        from langchain_core.language_models.fake import FakeListLLM
        return FakeListLLM(responses=["Mock response"])
    except ImportError:
        class SimpleFakeLLM:
            def invoke(self, prompt):
                class Response:
                    content = "Mock LLM response for testing"
                return Response()
        return SimpleFakeLLM()
```

### TIP #012: Proper Configuration Usage in LangGraph Nodes
**Category**: Development | **Severity**: High

**Problem Description**: Nodes that hardcode LLM model names instead of using the configuration system create inflexible applications.

**Solution Implementation**:
```python
# ❌ WRONG - Hardcoded model
def my_agent_node(state: OverallState, config: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # HARDCODED - BAD!
        temperature=0.1,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

# ✅ CORRECT - Use configuration
from agent.configuration import Configuration

def my_agent_node(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.answer_model,  # Use configured model
        temperature=0.1,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
```

### TIP #013: LangChain Message Object Handling in State
**Category**: Development | **Severity**: High  

**Problem Description**: In LangGraph runtime, messages in state are converted to LangChain message objects (HumanMessage, AIMessage) which don't have dictionary methods like `.get()`.

**Error Example**: `'HumanMessage' object has no attribute 'get'`

**Solution Implementation**:
```python
# ❌ WRONG - Treating messages as dictionaries
user_messages = [msg for msg in state.get("messages", []) if msg.get("role") == "user"]
latest_user_message = user_messages[-1]["content"] if user_messages else "Hola"

# ✅ CORRECT - Handle both dict and LangChain message objects  
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
```

### TIP #005: The Critical Gap Between Unit Tests and Real LLM Endpoint Testing
**Category**: Testing | **Severity**: Critical

**Problem Description**: Mock tests can pass while real API endpoints fail catastrophically. Unit tests use mocked dependencies, so they pass even when import statements are incorrect for the actual runtime environment.

**Solution Implementation**:
```bash
# STEP 1: Test graph loading in server context BEFORE unit tests
cd /backend_gen
python -c "from agent.graph import graph; print('✅ Graph loads successfully:', type(graph))"

# STEP 2: Start server and check logs for import errors
nohup langgraph dev > langgraph.log 2>&1 &
sleep 10

# CRITICAL: Check for import errors in server logs
if grep -q "ImportError\|ModuleNotFoundError" langgraph.log; then
    echo "❌ CRITICAL: Server has import errors"
    grep -A3 -B3 "Error" langgraph.log
    exit 1
fi
```

**Prevention Strategy**: Always test in this order:
1. **Graph import test** in server context first
2. **Server startup** with log monitoring for import errors  
3. **Real endpoint calls** with actual data payloads
4. **LLM response validation** in final state
5. **Only then** run unit tests as confirmation

## ITERATIVE BUSINESS CASE EXECUTION PROTOCOL

### Master Execution Loop

1. **Business Case Generation Phase**
   - Think creatively about this agentic business case
   - create examples for the llms, think about what can fail, extreme cases, etc.
   - Document the business case rationale and expected challenges
   - Create `/tasks/business_case.md` with detailed specification

2. **Implementation Phase**
   - Follow standard execution phases (0-3) for the business case
   - Apply lessons learned from `/docs/tips.md` proactively
   - Document new patterns and solutions discovered

3. **Error Learning Phase**
   - Every time an error is encountered and fixed, update `/docs/tips.md`
   - Follow the Enhanced Tips Format
   - Review existing tips before writing new code or tests

4. **Knowledge Accumulation Phase**
   - After creating the solution, update `/docs/patterns_learned.md`
   - Document successful architectural decisions
   - Note business case complexity vs implementation patterns

For each phase write a file in /tasks indicating the steps that have to be done and the instructions for each specific phase, and then follow what that file says. Include all the examples of code or tips for each phase in the file.

**NEVER use mock APIs, never, period.**

## USE CASE: Spanish NES Security Audit Assistant

The use case to implement is a conversational flow where you can ask for the next question and the agents help you interactively. For helping the user when working in an audit, there should be a .md file with the questions to fill, and the agents will help the user to fill the answers, they don't generate answers, the answers should be answered by the user, the agents can help the user to know what questions are still in the file without answer, or suggest how to write down the answers in a more formal way, for example if the question is how the company does backups and the user answer with a NAS, ask details to the user and at the end provide the user a better answer that "just a NAS". 

The agents should be experts in security audits in the NES national security standard in Spain, and the solution should ask the questions and fill the document in spanish.

The agents have to work with the .md file and help the user to answer the questions.

Create a .md file in the backend_gen folder with some example questions and use this path for the solution, dont need to ask the user, you have the path, read the document at the beginning of the solution and pass the questions to the agent, so it can start asking questions to the user. No need to use "keywords" like "siguiente pregunta" with the communication with the client, just use the llm conversation.

Make the lesser number of agents required. Use few agents, only the needed, better one agent with more tools than many agents. There will be an agent coordinator that will help the user to make work the use case, if it is possible only use this agent to make everything.

## Deployment

### Docker Build
```bash
docker build -t gemini-fullstack-langgraph -f Dockerfile .
```

### Production Server
```bash
GEMINI_API_KEY=<key> LANGSMITH_API_KEY=<key> docker-compose up
# Access at http://localhost:8123/app/
```

## Common Patterns

### Error Prevention
- Always test graph imports before server startup
- Use absolute imports in all graph files
- Validate environment variables before LLM calls
- Implement proper process cleanup in testing

### Agent Node Structure
Each agent node follows consistent patterns:
- Takes OverallState and RunnableConfig parameters
- Uses Configuration.from_runnable_config() for models
- Returns dict with state updates
- Handles errors gracefully with fallbacks

### Router vs Node Distinction
- Routers return string literals for graph navigation
- Nodes return dict state updates or None
- Only routers determine graph flow paths

This architecture enables systematic generation of complex multi-agent systems while maintaining consistency and reliability across different business domains.