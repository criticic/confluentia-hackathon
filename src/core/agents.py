"""
Core agent definitions and node implementations for the AI Copilot.
Enhanced with dynamic agent orchestration and specialized agents.
"""

from typing import TypedDict, Dict, Any, List, Optional
import logging
import json
import pandas as pd
from typing_extensions import Annotated
from langchain_core.messages import HumanMessage
from datetime import datetime

from src.prompts import (
    EXAMPLE_QUERY_GENERATOR_PROMPT,
    ORCHESTRATOR_PROMPT,
    ANALYZER_PROMPT,
    VALIDATOR_PROMPT,
    PLANNER_PROMPT,
    COORDINATOR_PROMPT,
    VISUALIZER_PROMPT,
)
from src.models.gemini import get_model
from src.tools.python_executor import execute_python_code
from src.core.memory import get_learned_logic # Import the memory function
from langgraph.graph.message import add_messages

logger = logging.getLogger("gabi.core.agents")

class AgentExecution(TypedDict):
    """Single agent execution record."""
    agent_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: str
    success: bool
    reasoning: str

class CopilotState(TypedDict):
    """Enhanced state definition for the flexible data copilot workflow."""
    # Core data
    question: str
    original_question: str
    dataframe_str: str
    column_info: str
    
    # Agent execution tracking
    agent_executions: List[AgentExecution]
    current_reasoning: str
    next_actions: List[str]
    completion_status: str  # "incomplete", "complete", "error"
    
    # Agent-specific outputs
    research_findings: str
    analysis_results: Dict[str, Any]
    validation_results: str
    planning_output: str
    coordinator_decision: str
    visualization_config: Dict[str, Any]  # Visualization configuration and data
    
    # Database context (optional)
    database_metadata: Optional[Dict[str, Any]]  # Contains DB schema, tables, etc.
    data_source_type: str  # "csv", "database", "mongodb"


model = get_model()
# A separate, more creative model for summarization and KPIs
creative_model = get_model(temperature=0.7)


def get_chat_history(state: CopilotState) -> str:
    """Formats the chat history for inclusion in prompts."""
    history_context = ""
    chat_history = state.get("chat_history")
    if chat_history:
        recent_history = chat_history[-3:]
        for exchange in recent_history:
            history_context += f"User: {exchange.get('question', '')}\n"
            response = exchange.get('response', {})
            # Add context from previous answers
            if isinstance(response, dict):
                if response.get('summary'):
                    history_context += f"Assistant: {response.get('summary')}\n"
                elif response.get('explanation'):
                    history_context += f"Assistant: {response.get('explanation')}\n"
    return history_context

def add_agent_execution(state: CopilotState, agent_name: str, input_data: Dict[str, Any], 
                       output_data: Dict[str, Any], success: bool, reasoning: str) -> Dict[str, Any]:
    """Helper function to add an agent execution record to the state."""
    execution: AgentExecution = {
        "agent_name": agent_name,
        "input_data": input_data,
        "output_data": output_data,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "reasoning": reasoning
    }
    
    current_executions = state.get("agent_executions", [])
    current_executions.append(execution)
    
    return {"agent_executions": current_executions}

def get_execution_history(state: CopilotState) -> str:
    """Format the agent execution history for prompts, including execution results."""
    executions = state.get("agent_executions", [])
    if not executions:
        return "No previous agent executions."
    
    history = "Previous Agent Executions:\n"
    for exec in executions[-5:]:  # Last 5 executions
        history += f"- {exec['agent_name']}: {exec['reasoning']}\n"
        
        # Include execution results if available
        output_data = exec.get('output_data', {})
        if 'execution' in output_data and output_data['execution']:
            execution_result = output_data['execution']
            if execution_result.get('output'):
                # Truncate long outputs but show key results
                output_text = str(execution_result['output'])
                if len(output_text) > 300:
                    output_text = output_text[:300] + "..."
                history += f"  EXECUTION OUTPUT: {output_text}\n"
            if execution_result.get('success'):
                history += "  STATUS: Execution successful\n"
        
        if not exec['success']:
            history += f"  ERROR: {output_data.get('error', 'Unknown error')}\n"
        
        history += "\n"
    
    return history

def generate_visualization_config(question: str, result_data: Dict[str, Any], python_code: str) -> Dict[str, Any]:
    """
    Generate visualization configuration based on analysis results.
    """
    try:
        if not result_data or result_data.get("type") != "dataframe":
            return {"chart_type": "none"}
        
        data = result_data.get("data", [])
        columns = result_data.get("columns", [])
        
        if not data or not columns:
            return {"chart_type": "table"}
        
        # Determine the best chart type based on data characteristics
        numeric_cols = []
        categorical_cols = []
        
        # Simple heuristic: check first row for data types
        if data:
            first_row = data[0]
            for col in columns:
                value = first_row.get(col)
                if isinstance(value, (int, float)):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
        
        # Determine chart type based on question and data structure
        question_lower = question.lower()
        
        if "trend" in question_lower or "over time" in question_lower:
            chart_type = "line"
        elif "compare" in question_lower or "vs" in question_lower or len(data) <= 20:
            chart_type = "bar"
        elif "distribution" in question_lower:
            chart_type = "histogram"
        elif "correlation" in question_lower and len(numeric_cols) >= 2:
            chart_type = "scatter"
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0 and len(data) <= 20:
            chart_type = "bar"
        elif len(numeric_cols) >= 2:
            chart_type = "scatter"
        else:
            chart_type = "table"
        
        config = {
            "chart_type": chart_type,
            "title": f"Analysis Results: {question[:50]}{'...' if len(question) > 50 else ''}",
            "data": data
        }
        
        # Set axes based on chart type and available columns
        if chart_type in ["bar", "line"] and categorical_cols and numeric_cols:
            config["x_axis"] = categorical_cols[0]
            config["y_axis"] = numeric_cols[0]
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            config["x_axis"] = numeric_cols[0]
            config["y_axis"] = numeric_cols[1]
            if len(categorical_cols) > 0:
                config["color_by"] = categorical_cols[0]
        elif chart_type == "histogram" and numeric_cols:
            config["x_axis"] = numeric_cols[0]
        elif chart_type == "pie" and categorical_cols and numeric_cols:
            config["x_axis"] = categorical_cols[0]  # names
            config["y_axis"] = numeric_cols[0]      # values
        
        logger.info(f"Generated visualization config: {chart_type}")
        return config
        
    except Exception as e:
        logger.error(f"Error generating visualization config: {e}")
        return {"chart_type": "table"}

# =====================
# DYNAMIC ORCHESTRATOR
# =====================

def orchestrator_node(state: CopilotState):
    """
    Controls the flow of the copilot pipeline by deciding which agent to call next.
    """
    logger.info("=== ORCHESTRATOR AGENT STARTED ===")
    question = state.get("question", "")
    original_question = state.get("original_question", question)
    current_reasoning = state.get("current_reasoning", "")
    completion_status = state.get("completion_status", "incomplete")
    execution_history = get_execution_history(state)
    column_info = state.get("column_info", "")
    agent_executions = state.get("agent_executions", [])
    
    # Circuit breaker: If we have too many executions, force completion
    if len(agent_executions) >= 20:
        logger.warning(f"Circuit breaker triggered - too many agent executions ({len(agent_executions)}), forcing completion")
        # Add execution record for the forced completion
        execution_update = add_agent_execution(
            state, "orchestrator",
            {"question": question, "forced": True},
            {"decision": "COMPLETE (circuit breaker)", "forced": True},
            True, "Forced completion due to excessive iterations"
        )
        return {
            "completion_status": "complete",
            "current_reasoning": "Task completed by circuit breaker to prevent infinite loop",
            "next_actions": [],
            "messages": [],
            **execution_update
        }
    
    # Check if we're already complete
    if completion_status == "complete":
        logger.info("Task already complete, ending workflow")
        return {"completion_status": "complete"}
    
    prompt = ORCHESTRATOR_PROMPT.format(
        original_question=original_question,
        question=question,
        current_reasoning=current_reasoning,
        completion_status=completion_status,
        execution_history=execution_history,
        column_info=column_info
    )

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    content = str(response.content) if hasattr(response, 'content') else str(response)
    
    logger.info(f"Orchestrator decision: {content}")
    logger.debug(f"Orchestrator full response: {content}")
    
    # Parse the orchestrator's decision
    if content.startswith("COMPLETE"):
        reasoning = content.split("|", 1)[1].strip() if "|" in content else "Task completed"
        logger.info(f"Orchestrator completed task: {reasoning}")
        logger.info(f"Orchestrator decision structured: action='COMPLETE', reasoning='{reasoning}'")
        return {
            "completion_status": "complete",
            "current_reasoning": reasoning,
            "next_actions": [],
            "messages": [response]
        }
    elif content.startswith("CALL:"):
        parts = content[5:].split("|", 1)
        next_agent = parts[0].strip()
        reasoning = parts[1].strip() if len(parts) > 1 else "No reasoning provided"
        
        # Check for infinite loops - if we've called the same agent too many times recently
        recent_executions = agent_executions[-10:] if len(agent_executions) >= 10 else agent_executions
        recent_agent_calls = [ex.get("agent_name") for ex in recent_executions]
        
        # Smart feedback handling based on validation results
        if next_agent == "validator":
            # Check if validator was just called and failed
            last_validator_result = None
            for ex in reversed(agent_executions):
                if ex.get("agent_name") == "validator":
                    last_validator_result = ex.get("result", "")
                    break
            
            # If validator just failed, don't call it again - take corrective action instead
            if last_validator_result and ("NEEDS WORK" in last_validator_result or "NEEDS_IMPROVEMENT" in last_validator_result):
                logger.info("Validator previously failed, redirecting to analyzer to fix issues")
                next_agent = "analyzer"
                reasoning = "Validator identified issues, redirecting to analyzer to incorporate feedback and fix problems"
            elif recent_agent_calls.count("validator") >= 2:
                logger.warning("Validator called multiple times recently, forcing coordinator to break loop")
                next_agent = "coordinator"
                reasoning = "Moving to coordinator due to repeated validation attempts"
        
        # Prevent infinite analyzer loops - if analyzer has been called multiple times and is succeeding, consider visualization
        elif next_agent == "analyzer":
            analyzer_count = recent_agent_calls.count("analyzer")
            if analyzer_count >= 3:
                # Check if recent analyzer calls were successful
                recent_analyzer_successes = 0
                recent_analyzer_data = False
                for ex in reversed(agent_executions[-5:]):
                    if ex.get("agent_name") == "analyzer" and ex.get("success", False):
                        recent_analyzer_successes += 1
                        # Check if analyzer produced data suitable for visualization
                        if "execution" in str(ex.get("output_data", {})) and "dataframe" in str(ex.get("output_data", {})):
                            recent_analyzer_data = True
                
                if recent_analyzer_successes >= 2:
                    # If we have data and haven't visualized yet, suggest visualization
                    if recent_analyzer_data and "visualizer" not in recent_agent_calls:
                        logger.info(f"Analyzer produced data results, suggesting visualization before completion")
                        next_agent = "visualizer"
                        reasoning = "Analyzer has produced data results - creating visualization to help understand the findings"
                    else:
                        logger.warning(f"Analyzer called {analyzer_count} times recently with {recent_analyzer_successes} successes, forcing coordinator to complete task")
                        next_agent = "coordinator"
                        reasoning = "Moving to coordinator - analyzer has successfully completed analysis multiple times"
        
        logger.info(f"Orchestrator calling: {next_agent} - {reasoning}")
        logger.info(f"Orchestrator decision structured: action='CALL', next_agent='{next_agent}', reasoning='{reasoning}'")
        
        # Add execution record for orchestrator decision
        execution_update = add_agent_execution(
            state, "orchestrator",
            {"question": question, "next_agent": next_agent},
            {"decision": f"CALL: {next_agent}", "reasoning": reasoning},
            True, f"Routing to {next_agent}"
        )
        
        return {
            "next_actions": [next_agent],
            "current_reasoning": reasoning,
            "completion_status": "incomplete",
            "messages": [response],
            **execution_update
        }
    else:
        # Check if we should suggest visualization before falling back
        # Look for recent successful analysis that might benefit from visualization
        has_data_results = False
        has_visualizer_run = False
        
        for ex in agent_executions:
            if ex.get("agent_name") == "analyzer" and ex.get("success", False):
                output_data = ex.get("output_data", {})
                if "execution" in str(output_data) and "dataframe" in str(output_data):
                    has_data_results = True
            elif ex.get("agent_name") == "visualizer":
                has_visualizer_run = True
        
        # If we have data results but no visualization, suggest that
        if has_data_results and not has_visualizer_run:
            logger.info("Found data results without visualization, suggesting visualizer")
            execution_update = add_agent_execution(
                state, "orchestrator",
                {"question": question, "fallback": True, "suggested": "visualizer"},
                {"decision": "CALL: visualizer", "reasoning": "Found analysis results that would benefit from visualization"},
                True, "Suggesting visualization for analysis results"
            )
            return {
                "next_actions": ["visualizer"],
                "current_reasoning": "Found analysis results that would benefit from visualization",
                "completion_status": "incomplete",
                "messages": [response],
                **execution_update
            }
        
        # Fallback to analyzer if unclear
        logger.warning(f"Unclear orchestrator response, defaulting to analyzer: {content}")
        execution_update = add_agent_execution(
            state, "orchestrator",
            {"question": question, "fallback": True},
            {"decision": "CALL: analyzer", "reasoning": "Fallback due to unclear response"},
            True, "Fallback routing to analyzer"
        )
        return {
            "next_actions": ["analyzer"],
            "current_reasoning": "Fallback to analyzer due to unclear orchestrator response",
            "completion_status": "incomplete",
            "messages": [response],
            **execution_update
        }

# ========================
# HELPER FUNCTIONS
# ========================

def convert_sql_to_pandas(sql_code: str, question: str) -> str:
    """
    Convert simple SQL queries to pandas equivalent code.
    This is a basic conversion for common patterns.
    """
    try:
        sql_upper = sql_code.upper().strip()
        
        # Handle simple SELECT with ORDER BY and LIMIT patterns
        if "SELECT" in sql_upper and "ORDER BY" in sql_upper and "LIMIT" in sql_upper:
            # Pattern for top N queries like: SELECT col1, col2 FROM table ORDER BY col DESC LIMIT N
            
            # Extract column names (simple pattern matching)
            if "model_name" in sql_code.lower() and "model_base_price" in sql_code.lower():
                if "DESC" in sql_upper:
                    return "result = df.nlargest(5, 'model_base_price')[['model_name', 'model_base_price']]"
                else:
                    return "result = df.nsmallest(5, 'model_base_price')[['model_name', 'model_base_price']]"
            
            # Generic pattern for top N by a column
            if "ORDER BY" in sql_upper and "DESC" in sql_upper:
                return f"# Converted from SQL: {sql_code}\n# Finding top records by value\nresult = df.head(5)\nprint('SQL to pandas conversion attempted, showing first 5 rows')"
        
        # Fallback: generate basic pandas code based on the question
        if "top" in question.lower() and "price" in question.lower():
            return "result = df.nlargest(5, df.select_dtypes(include=['number']).columns[0])"
        
        # Default fallback
        return "result = df.head(10)\nprint('SQL code detected but could not convert automatically. Showing first 10 rows.')"
        
    except Exception as e:
        logger.error(f"Error converting SQL to pandas: {e}")
        return "result = df.head(10)\nprint('Error converting SQL. Showing first 10 rows.')"

# ========================
# SPECIALIZED AGENTS
# ========================

def analyzer_node(state: CopilotState):
    """
    Performs specific data analysis and generates Python code or SQL/MongoDB queries when needed.
    Enhanced to work with databases and multiple data sources.
    """
    logger.info("=== ANALYZER AGENT STARTED ===")
    question = state.get("question", "")
    column_info = state.get("column_info", "")
    dataframe_str = state.get("dataframe_str", "")
    research_findings = state.get("research_findings", "")
    execution_history = get_execution_history(state)
    database_metadata = state.get("database_metadata", {})
    data_source_type = state.get("data_source_type", "csv")

    logger.info(f"Analyzer inputs: data_source_type='{data_source_type}', has_database_metadata={bool(database_metadata)}")
    logger.info(f"Database metadata keys: {list(database_metadata.keys()) if database_metadata else 'None'}")

    # --- MEMORY ENHANCEMENT ---
    # Check for learned logic from previous user interactions
    learned_code = get_learned_logic(question)
    learned_logic_prompt = ""
    if learned_code:
        learned_logic_prompt = f"""
USER PREFERENCE: For a similar question in the past, a user preferred the following code.
Consider using this logic if it is applicable to the current request.
```python
{learned_code}
```
"""
    
    # Check for recent validator feedback to incorporate improvements
    validator_feedback = ""
    agent_executions = state.get("agent_executions", [])
    for ex in reversed(agent_executions):
        if ex.get("agent_name") == "validator" and ex.get("result"):
            validator_feedback = ex.get("result", "")
            break
    
    # Build context based on data source type
    data_context = ""
    if data_source_type == "database" and database_metadata:
        tables_info = database_metadata.get("tables", {})
        db_type = database_metadata.get("database_type", "unknown")
        
        data_context = f"""
DATABASE CONTEXT:
- Database Type: {db_type}
- Available Tables: {list(tables_info.keys())}

TABLE SCHEMAS:
"""
        for table_name, table_info in tables_info.items():
            columns = table_info.get("columns", [])
            types = table_info.get("column_types", [])
            row_count = table_info.get("row_count", "Unknown")
            
            data_context += f"""
- {table_name} ({row_count} rows):
  Columns: {', '.join(columns)}
"""
        
        data_context += """
You can generate SQL queries to analyze this database. Use proper JOIN operations when needed.
"""
    
    elif data_source_type == "mongodb" and database_metadata:
        collections_info = database_metadata.get("collections", {})
        
        data_context = f"""
MONGODB CONTEXT:
- Database Type: MongoDB
- Available Collections: {list(collections_info.keys())}

COLLECTION SCHEMAS:
"""
        for collection_name, collection_info in collections_info.items():
            fields = collection_info.get("fields", [])
            doc_count = collection_info.get("document_count", "Unknown")
            
            data_context += f"""
- {collection_name} ({doc_count} documents):
  Fields: {', '.join(fields)}
"""
        
        data_context += """
You can generate MongoDB queries using PyMongo syntax to analyze this database.
"""
    
    else:
        data_context = f"""
CSV/DATAFRAME CONTEXT:
Dataset Schema: {column_info}
Sample Data:
{dataframe_str}
"""

    # Enhanced prompt with database/source context
    if data_source_type == "database":
        code_instruction = "Generate SQL queries to extract and analyze data from the database tables"
        code_type = "SQL query"
    elif data_source_type == "mongodb":
        code_instruction = "Generate PyMongo queries to extract and analyze data from the MongoDB collections"
        code_type = "MongoDB query"
    else:
        code_instruction = "Generate Python pandas code for analysis (the DataFrame is available as 'df')"
        code_type = "Python code"

    enhanced_prompt = f"""You are a data analysis specialist. Your job is to perform specific analysis to answer the user's question.

Question: {question}

{data_context}

Research Context:
{research_findings}

Previous Work:
{execution_history}

{learned_logic_prompt}

DATA SOURCE TYPE: {data_source_type.upper()}

IMPORTANT: {code_instruction}

CRITICAL GUIDELINES:
- This is {data_source_type} data, so you MUST generate {code_type}
- Always assign your final result to a variable called 'result'
- For Python code: Use pandas operations with the 'df' DataFrame
- For SQL: Write proper SQL queries that can be executed against the database
- For MongoDB: Use PyMongo syntax with proper aggregation pipelines

Your response should be structured as:
APPROACH: [describe your analytical approach]
CODE: [{code_type}]
ANALYSIS: [your findings and insights]

IMPORTANT: Do NOT include fake or example data in the RESULT section. The system will execute your code and display real results automatically.

The system will automatically generate visualizations for tabular results.
"""
    
    if validator_feedback and ("NEEDS" in validator_feedback or "ISSUES:" in validator_feedback):
        enhanced_prompt += f"\n\nIMPORTANT - VALIDATOR FEEDBACK TO ADDRESS:\n{validator_feedback}\n\nPlease incorporate this feedback and address any issues mentioned above."
    
    prompt = enhanced_prompt

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    content = str(response.content) if hasattr(response, 'content') else str(response)
    
    logger.info(f"Analyzer response: {content[:400]}...")
    logger.debug(f"Analyzer full response: {content}")
    
    # Parse the response to extract code if present
    analysis_result: Dict[str, Any] = {"type": "analysis", "content": content}
    python_code = ""
    visualization_config = {}
    
    # More robust code extraction
    if "CODE:" in content:
        try:
            # Find the code section
            code_start = content.find("CODE:") + 5
            
            # Look for the next section marker or end of content
            next_markers = ["ANALYSIS:", "RESULT:", "APPROACH:", "\n\n#", "\n---"]
            code_end = len(content)
            for marker in next_markers:
                marker_pos = content.find(marker, code_start)
                if marker_pos != -1:
                    code_end = min(code_end, marker_pos)
            
            code_section = content[code_start:code_end].strip()
            
            # Clean up the code
            if code_section and code_section.lower() != "none":
                # Remove markdown code blocks if present
                if code_section.startswith("```sql") or code_section.startswith("```python"):
                    code_section = code_section[6:]
                elif code_section.startswith("```"):
                    code_section = code_section[3:]
                if code_section.endswith("```"):
                    code_section = code_section[:-3]
                
                extracted_code = code_section.strip()
                
                # Validate code type based on data source
                if data_source_type == "csv" or data_source_type == "dataframe":
                    # For CSV/DataFrame, we expect Python code
                    if extracted_code and any(sql_keyword in extracted_code.upper() for sql_keyword in ["SELECT", "FROM", "WHERE", "ORDER BY", "GROUP BY", "INSERT", "UPDATE", "DELETE"]):
                        logger.warning("SQL code detected for CSV/DataFrame data source. Converting to Python...")
                        # Try to convert simple SQL to pandas equivalent
                        python_code = convert_sql_to_pandas(extracted_code, question)
                    else:
                        python_code = extracted_code
                else:
                    # For database sources, SQL is expected
                    python_code = extracted_code
                
                # Validate that the code is not empty or just whitespace
                if python_code and len(python_code.strip()) > 0:
                    logger.info(f"Extracted code for {data_source_type}: {python_code[:100]}...")
                    logger.info(f"=== DETERMINING EXECUTION PATH ===")
                    logger.info(f"Data source type: {data_source_type}")
                    logger.info(f"Has database metadata: {bool(database_metadata)}")
                    
                    execution_result = None
                    # Execute Python code for CSV/DataFrame sources
                    if data_source_type == "csv" or data_source_type == "dataframe":
                        logger.info(f"=== TAKING CSV/DATAFRAME EXECUTION PATH ===")
                        
                        # Check if this is actually SQL code being misclassified
                        if any(sql_keyword in python_code.upper() for sql_keyword in ["SELECT", "FROM", "WHERE", "ORDER BY", "GROUP BY"]):
                            logger.warning(f"SQL code detected in CSV mode! Forcing database execution.")
                            # Force database execution
                            import os
                            import sqlite3
                            import pandas as pd
                            
                            local_db_path = os.path.join(os.getcwd(), "data.db")
                            if os.path.exists(local_db_path):
                                logger.info(f"=== FORCING SQL EXECUTION ON LOCAL DATABASE ===")
                                try:
                                    logger.info(f"Executing SQL query: {python_code}")
                                    conn = sqlite3.connect(local_db_path)
                                    result_df = pd.read_sql_query(python_code, conn)
                                    conn.close()
                                    logger.info(f"SQL execution completed. Result shape: {result_df.shape}")
                                    logger.info(f"SQL result preview: {result_df.head().to_dict('records')}")
                                    
                                    if result_df is not None and not result_df.empty:
                                        execution_result = {
                                            "success": True,
                                            "result": {
                                                "type": "dataframe",
                                                "data": result_df.to_dict('records'),
                                                "columns": list(result_df.columns),
                                                "shape": result_df.shape
                                            },
                                            "message": f"Successfully executed SQL query on local database"
                                        }
                                        analysis_result["execution"] = execution_result
                                        analysis_result["type"] = "sql_analysis"
                                        logger.info(f"Forced SQL execution successful: {len(result_df)} rows returned")
                                    else:
                                        execution_result = {
                                            "success": False,
                                            "error": "SQL query returned no results"
                                        }
                                        analysis_result["execution"] = execution_result
                                except Exception as e:
                                    logger.error(f"Forced SQL execution failed: {str(e)}")
                                    # Fall back to storing SQL code
                                    analysis_result["sql_code"] = python_code
                                    analysis_result["type"] = "sql_generated"
                                    analysis_result["message"] = f"SQL query generated but execution failed: {str(e)}"
                            else:
                                logger.warning("No local database found for forced SQL execution")
                                # Fall back to storing SQL code
                                analysis_result["sql_code"] = python_code
                                analysis_result["type"] = "sql_generated"
                                analysis_result["message"] = "SQL query generated but no database available"
                        else:
                            # Normal CSV/DataFrame processing
                            execution_result = execute_python_code(python_code, dataframe_str)
                            analysis_result["execution"] = execution_result
                            analysis_result["type"] = "code_analysis"
                    
                    # Execute SQL for database sources
                    elif data_source_type == "database" and database_metadata:
                        logger.info(f"=== DATABASE EXECUTION PATH STARTED ===")
                        try:
                            from src.utils.db_connector import connect_and_query_sql
                            import streamlit as st
                            
                            # Get database connection details from session state
                            db_type = database_metadata.get("database_type", "").lower()
                            conn_details = st.session_state.get("db_connection_details", {})
                            
                            logger.info(f"Database execution details: db_type='{db_type}', conn_details_available={bool(conn_details)}")
                            
                            if db_type and conn_details:
                                logger.info(f"Executing SQL query on {db_type} database")
                                result_df = connect_and_query_sql(db_type, conn_details, python_code)
                                
                                if result_df is not None and not result_df.empty:
                                    # Convert DataFrame to execution result format
                                    execution_result = {
                                        "success": True,
                                        "result": {
                                            "type": "dataframe",
                                            "data": result_df.to_dict('records'),
                                            "columns": result_df.columns.tolist(),
                                            "shape": result_df.shape,
                                            "dtypes": result_df.dtypes.astype(str).to_dict()
                                        },
                                        "output": f"Query returned {len(result_df)} rows"
                                    }
                                    analysis_result["execution"] = execution_result
                                    analysis_result["type"] = "sql_analysis"
                                    logger.info(f"SQL execution successful: {len(result_df)} rows returned")
                                else:
                                    execution_result = {
                                        "success": False,
                                        "error": "Query returned no results or failed to execute"
                                    }
                                    analysis_result["execution"] = execution_result
                            else:
                                logger.warning(f"Missing database connection details. db_type={db_type}, conn_details available={bool(conn_details)}")
                                logger.info(f"=== ATTEMPTING SQLITE FALLBACK ===")
                                
                                # Try to use local data.db as fallback for SQLite
                                if db_type == "sqlite" and not conn_details:
                                    import os
                                    import sqlite3
                                    import pandas as pd
                                    
                                    local_db_path = os.path.join(os.getcwd(), "data.db")
                                    logger.info(f"Checking for local database at: {local_db_path}")
                                    if os.path.exists(local_db_path):
                                        logger.info(f"=== LOCAL DATABASE FOUND, EXECUTING QUERY ===")
                                        logger.info(f"Attempting to use local database: {local_db_path}")
                                        try:
                                            # Direct SQLite connection without Streamlit dependencies
                                            logger.info(f"Executing SQL query: {python_code}")
                                            conn = sqlite3.connect(local_db_path)
                                            result_df = pd.read_sql_query(python_code, conn)
                                            conn.close()
                                            logger.info(f"SQL execution completed. Result shape: {result_df.shape}")
                                            logger.info(f"SQL result preview: {result_df.head().to_dict('records')}")
                                            
                                            if result_df is not None and not result_df.empty:
                                                execution_result = {
                                                    "success": True,
                                                    "result": {
                                                        "type": "dataframe",
                                                        "data": result_df.to_dict('records'),
                                                        "columns": list(result_df.columns),
                                                        "shape": result_df.shape
                                                    },
                                                    "message": f"Successfully executed query using local database: {local_db_path}"
                                                }
                                                analysis_result["execution"] = execution_result
                                                logger.info(f"Local SQLite execution successful: {len(result_df)} rows returned")
                                            else:
                                                execution_result = {
                                                    "success": False,
                                                    "error": "Query returned no results from local database"
                                                }
                                                analysis_result["execution"] = execution_result
                                        except Exception as local_e:
                                            logger.error(f"Local database fallback failed: {str(local_e)}")
                                            # Store the SQL for later visualization
                                            analysis_result["sql_code"] = python_code
                                            analysis_result["type"] = "sql_generated"
                                            analysis_result["message"] = f"SQL query generated. Local database connection failed: {str(local_e)}"
                                    else:
                                        # Store the SQL for later visualization even if we can't execute it
                                        analysis_result["sql_code"] = python_code
                                        analysis_result["type"] = "sql_generated"
                                        analysis_result["message"] = "SQL query generated successfully. Please connect to your database to execute this query."
                                else:
                                    # Store the SQL for later visualization even if we can't execute it
                                    analysis_result["sql_code"] = python_code
                                    analysis_result["type"] = "sql_generated"
                                    analysis_result["message"] = f"SQL query generated for {db_type} database. Please provide connection details to execute this query."
                        except Exception as e:
                            logger.error(f"Error executing SQL query: {e}")
                            execution_result = {
                                "success": False,
                                "error": f"SQL execution failed: {str(e)}"
                            }
                            analysis_result["execution"] = execution_result
                    
                    # Execute MongoDB queries for mongodb sources  
                    elif data_source_type == "mongodb" and database_metadata:
                        try:
                            from src.utils.db_connector import connect_and_query_mongo
                            import streamlit as st
                            
                            conn_details = st.session_state.get("db_connection_details", {})
                            collection_name = st.session_state.get("selected_collection", "")
                            
                            if conn_details and collection_name:
                                logger.info(f"Executing MongoDB query on collection {collection_name}")
                                result_df = connect_and_query_mongo(conn_details, collection_name, python_code)
                                
                                if result_df is not None and not result_df.empty:
                                    execution_result = {
                                        "success": True,
                                        "result": {
                                            "type": "dataframe",
                                            "data": result_df.to_dict('records'),
                                            "columns": result_df.columns.tolist(),
                                            "shape": result_df.shape,
                                            "dtypes": result_df.dtypes.astype(str).to_dict()
                                        },
                                        "output": f"Query returned {len(result_df)} documents"
                                    }
                                    analysis_result["execution"] = execution_result
                                    analysis_result["type"] = "mongodb_analysis"
                                    logger.info(f"MongoDB execution successful: {len(result_df)} documents returned")
                                else:
                                    execution_result = {
                                        "success": False,
                                        "error": "Query returned no results or failed to execute"
                                    }
                                    analysis_result["execution"] = execution_result
                            else:
                                logger.warning(f"Missing MongoDB connection details. conn_details available={bool(conn_details)}, collection={collection_name}")
                        except Exception as e:
                            logger.error(f"Error executing MongoDB query: {e}")
                            execution_result = {
                                "success": False,
                                "error": f"MongoDB execution failed: {str(e)}"
                            }
                            analysis_result["execution"] = execution_result
                    
                    # Generate visualization if execution was successful and returned a DataFrame
                    if execution_result and execution_result.get("success") and execution_result.get("result"):
                        result_data = execution_result.get("result")
                        if isinstance(result_data, dict) and result_data.get("type") == "dataframe":
                            visualization_config = generate_visualization_config(
                                question, result_data, python_code
                            )
                else:
                    logger.warning("Extracted code was empty after cleaning")
                    python_code = ""
        except Exception as e:
            logger.error(f"Error extracting code from analyzer response: {e}")
            python_code = ""
    
    logger.info("Analyzer completed analysis")
    
    # Log structured analysis results
    analysis_type = analysis_result.get("type", "unknown")
    has_execution = "execution" in analysis_result
    execution_success = analysis_result.get("execution", {}).get("success", False) if has_execution else False
    code_generated = bool(python_code and len(python_code.strip()) > 0)
    
    logger.info(f"Analyzer structured results: type='{analysis_type}', has_execution={has_execution}, execution_success={execution_success}, code_generated={code_generated}")
    
    if has_execution and execution_success:
        result_shape = analysis_result.get("execution", {}).get("result", {}).get("shape", "unknown")
        logger.info(f"Analyzer execution details: result_shape={result_shape}")
    
    # Add execution record
    execution_update = add_agent_execution(
        state, "analyzer",
        {"question": question, "approach": "data analysis"},
        analysis_result,
        True, "Performed data analysis"
    )
    
    return {
        "analysis_results": analysis_result,
        "python_code": python_code,
        "execution_result": analysis_result.get("execution", {}),
        "visualization_config": visualization_config,
        "messages": [response],
        **execution_update
    }

def validator_node(state: CopilotState):
    """
    Validates analysis results and checks for errors or improvements.
    """
    logger.info("=== VALIDATOR AGENT STARTED ===")
    question = state.get("question", "")
    analysis_results = state.get("analysis_results", {})
    execution_result = state.get("execution_result", {})
    execution_history = get_execution_history(state)
    
    prompt = VALIDATOR_PROMPT.format(
        question=question,
        analysis_results=analysis_results,
        execution_result=execution_result,
        execution_history=execution_history
    )

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    content = str(response.content) if hasattr(response, 'content') else str(response)
    
    # Log the validator response for debugging
    logger.info(f"Validator response: {content[:500]}...")
    logger.debug(f"Validator full response: {content}")
    
    # Parse validation status - be more flexible with parsing
    is_valid = (
        "STATUS: VALID" in content or 
        "VALID" in content.upper() or
        ("NEEDS_IMPROVEMENT" not in content and "ERROR" not in content and len(content) > 50)
    )
    
    logger.info(f"Validator parsed result - is_valid: {is_valid}")
    
    # Override: If we keep getting stuck in validation loop, force validation to pass
    current_executions = state.get("agent_executions", [])
    validator_count = sum(1 for ex in current_executions if ex.get("agent_name") == "validator")
    
    # Log structured validator results
    forced_validation = validator_count >= 3 and not is_valid
    logger.info(f"Validator structured results: is_valid={is_valid}, validator_run_count={validator_count}, forced_validation={forced_validation}")
    
    if validator_count >= 3 and not is_valid:
        logger.warning("Validator has run 3+ times, forcing validation to pass to prevent infinite loop")
        is_valid = True
        content += "\n\nSTATUS: VALID (forced after multiple validation attempts)"
    
    logger.info(f"Validator completed: {'VALID' if is_valid else 'NEEDS WORK'}")
    
    # Add execution record
    execution_update = add_agent_execution(
        state, "validator",
        {"analysis_results": analysis_results},
        {"validation": content, "is_valid": is_valid},
        True, "Validated analysis results"
    )
    
    return {
        "validation_results": content,
        "messages": [response],
        **execution_update
    }

def planner_node(state: CopilotState):
    """
    Creates step-by-step plans for complex questions and breaks down tasks.
    """
    logger.info("=== PLANNER AGENT STARTED ===")
    question = state.get("question", "")
    original_question = state.get("original_question", question)
    column_info = state.get("column_info", "")
    execution_history = get_execution_history(state)
    
    prompt = PLANNER_PROMPT.format(
        original_question=original_question,
        question=question,
        column_info=column_info,
        execution_history=execution_history
    )

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    content = str(response.content) if hasattr(response, 'content') else str(response)
    
    logger.info(f"Planner response: {content[:300]}...")
    logger.debug(f"Planner full response: {content}")
    logger.info("Planner created analysis plan")
    
    # Add execution record
    execution_update = add_agent_execution(
        state, "planner",
        {"question": question, "context": "strategic planning"},
        {"plan": content},
        True, "Created strategic analysis plan"
    )
    
    return {
        "planning_output": content,
        "messages": [response],
        **execution_update
    }

def coordinator_node(state: CopilotState):
    """
    Makes final decisions about task completion and synthesizes results.
    """
    logger.info("=== COORDINATOR AGENT STARTED ===")
    question = state.get("question", "")
    original_question = state.get("original_question", question)
    research_findings = state.get("research_findings", "")
    analysis_results = state.get("analysis_results", {})
    validation_results = state.get("validation_results", "")
    execution_history = get_execution_history(state)
    viz_config = state.get("visualization_config", {})
    
    # Log what we have for debugging
    logger.info(f"Coordinator inputs - Research: {bool(research_findings)}, Analysis: {bool(analysis_results)}, Validation: {bool(validation_results)}")
    logger.info(f"Validation results content: {str(validation_results)[:200]}...")
    logger.info(f"Execution history length: {len(execution_history)}")
    logger.info(f"Coordinator viz_config: {viz_config}")
    logger.info(f"Coordinator viz_config type: {type(viz_config)}")
    logger.info(f"Coordinator viz_config keys: {list(viz_config.keys()) if isinstance(viz_config, dict) else 'N/A'}")
    
    prompt = COORDINATOR_PROMPT.format(
        original_question=original_question,
        question=question,
        research_findings=research_findings,
        analysis_results=analysis_results,
        validation_results=validation_results,
        execution_history=execution_history
    )

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    content = str(response.content) if hasattr(response, 'content') else str(response)
    
    logger.info(f"Coordinator response: {content[:400]}...")
    logger.debug(f"Coordinator full response: {content}")
    
    # Parse coordinator decision with multiple possible formats
    is_complete = (
        "DECISION: COMPLETE" in content or 
        "COMPLETE" in content.upper()
    )
    
    # Check if we have sufficient results to force completion
    has_validation = validation_results and len(str(validation_results)) > 20
    validation_passed = has_validation and ("VALID" in str(validation_results) or "valid" in str(validation_results))
    has_analysis = analysis_results and len(str(analysis_results)) > 50
    has_research = research_findings and len(str(research_findings)) > 50
    
    # Force completion if we have sufficient results
    if (has_research and has_analysis and has_validation and 
        not is_complete and len(execution_history) > 3):
        logger.info("Forcing completion due to sufficient results and execution history")
        is_complete = True
        content += "\n\nDECISION: COMPLETE (auto-completed due to sufficient analysis)"
    
    # Also force completion if validation passed and we have analysis
    elif validation_passed and has_analysis and not is_complete:
        logger.info("Forcing completion due to validation passed and analysis complete")
        is_complete = True
        content += "\n\nDECISION: COMPLETE (auto-completed - validation passed)"
    
    logger.info(f"Coordinator decision: {'COMPLETE' if is_complete else 'CONTINUE'}")
    
    # Log structured coordinator analysis
    logger.info(f"Coordinator structured results: decision={'COMPLETE' if is_complete else 'CONTINUE'}, has_research={has_research}, has_analysis={has_analysis}, has_validation={has_validation}, validation_passed={validation_passed}, execution_count={len(execution_history)}")
    
    # Add execution record
    execution_update = add_agent_execution(
        state, "coordinator",
        {"all_results": "synthesis"},
        {"decision": content, "is_complete": is_complete},
        True, "Coordinated final decision"
    )
    
    completion_status = "complete" if is_complete else "incomplete"
    
    # If completing, ensure we have the final visualization if available
    final_viz_config = viz_config if viz_config else {}
    
    return {
        "coordinator_decision": content,
        "completion_status": completion_status,
        "visualization_config": final_viz_config,
        "messages": [response],
        **execution_update
    }


def generate_dynamic_example_queries(dataframe: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Generate dynamic example queries based on the uploaded DataFrame schema and sample data.
    
    Args:
        dataframe: The uploaded pandas DataFrame
        
    Returns:
        Dictionary with categories as keys and lists of example queries as values
    """
    logger.info("Generating dynamic example queries based on uploaded data")
    
    try:
        # Get column information
        column_info = "\n".join([f"- {col} ({dtype})" for col, dtype in dataframe.dtypes.items()])
        
        # Get sample data (first 3 rows as a more compact representation)
        sample_data = dataframe.head(3).to_dict('records')
        sample_data_str = str(sample_data)
        
        prompt = EXAMPLE_QUERY_GENERATOR_PROMPT.format(
            column_info=column_info,
            sample_data=sample_data_str
        )
        
        messages = [HumanMessage(content=prompt)]
        response = creative_model.invoke(messages)
        content = str(response.content) if hasattr(response, 'content') else str(response)
        
        # Extract JSON from the response
        try:
            json_start = content.find("```json") + 7 if "```json" in content else content.find("{")
            json_end = content.rfind("```") if "```json" in content else content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end].strip()
                dynamic_queries = json.loads(json_str)
                logger.info("Successfully generated dynamic example queries")
                return dynamic_queries
            else:
                raise ValueError("Could not find valid JSON in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing dynamic query JSON: {e}")
            # Fallback to default queries if generation fails
            from src.config import EXAMPLE_QUERIES
            return EXAMPLE_QUERIES
            
    except Exception as e:
        logger.error(f"Error generating dynamic example queries: {e}")
        # Fallback to default queries if generation fails
        from src.config import EXAMPLE_QUERIES
        return EXAMPLE_QUERIES


def generate_database_specific_queries(db_metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generate dynamic example queries based on database metadata and table schemas.
    
    Args:
        db_metadata: Dictionary containing database metadata with table info
        
    Returns:
        Dictionary with categories as keys and lists of example queries as values
    """
    logger.info("Generating database-specific example queries based on schema")
    
    try:
        # Extract meaningful information about the database
        tables_info = db_metadata.get("tables", {})
        db_type = db_metadata.get("database_type", "unknown")
        
        # Build context for the AI
        schema_info = []
        for table_name, table_data in tables_info.items():
            columns = table_data.get("columns", [])
            column_types = table_data.get("column_types", [])
            row_count = table_data.get("row_count", 0)
            sample_data = table_data.get("sample_data", [])
            
            schema_info.append(f"""
Table: {table_name}
- Rows: {row_count}
- Columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}
- Types: {', '.join(set(column_types[:5]))}
- Sample: {str(sample_data[:2]) if sample_data else 'No sample data'}
""")
        
        schema_context = "\n".join(schema_info)
        
        prompt = f"""You are an expert data analyst. Based on the following database schema, generate specific, actionable natural language questions and requests that would be valuable for exploring this data.

Database Type: {db_type}
Schema Information:
{schema_context}

Generate a JSON response with 3 categories of natural language queries. Make the queries specific to the actual table names and business context, but phrase them as questions or requests a business user would ask - NOT as SQL code. The system will generate SQL later.

Format:
{{
  "Data Exploration": [
    "Show me the top 10 records from [table_name] with the highest values",
    "What are the unique categories in [table_name]?",
    "Display a sample of data from [table_name] to understand its structure"
  ],
  "Business Insights": [
    "Which [category/type] has the most activity or highest values?",
    "Show me trends over time in [table_name]",
    "Find patterns and correlations between different tables"
  ],
  "Data Quality": [
    "Check for missing or null values across all tables",
    "Identify any duplicate records in [table_name]",
    "Show me data completeness statistics for each table"
  ]
}}

Use the actual table names from the schema, but phrase everything as natural language questions or requests. Focus on what business users would want to know, not technical SQL queries."""

        messages = [HumanMessage(content=prompt)]
        response = creative_model.invoke(messages)
        content = str(response.content) if hasattr(response, 'content') else str(response)
        
        # Extract JSON from the response
        try:
            json_start = content.find("```json") + 7 if "```json" in content else content.find("{")
            json_end = content.rfind("```") if "```json" in content else content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end].strip()
                dynamic_queries = json.loads(json_str)
                logger.info("Successfully generated database-specific example queries")
                return dynamic_queries
            else:
                raise ValueError("Could not find valid JSON in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing database query JSON: {e}")
            # Fallback to basic database queries
            table_names = list(tables_info.keys())
            first_table = table_names[0] if table_names else "your data"
            
            return {
                "Data Exploration": [
                    f"Show me a sample of data from {first_table}",
                    f"What is the total number of records in {first_table}?",
                    f"Describe the structure and columns of {first_table}"
                ],
                "Business Insights": [
                    f"What are the key patterns and trends in {first_table}?",
                    f"Show me the distribution of values in {first_table}",
                    f"Identify the most important insights from {first_table}"
                ],
                "Data Quality": [
                    f"Check for any missing or incomplete data in {first_table}",
                    f"Are there any duplicate records in {first_table}?",
                    f"Analyze the overall data quality across all tables"
                ]
            }
            
    except Exception as e:
        logger.error(f"Error generating database-specific queries: {e}")
        # Fallback to default queries
        from src.config import EXAMPLE_QUERIES
        return EXAMPLE_QUERIES


def visualizer_node(state: CopilotState):
    """
    Creates Plotly visualizations by having the agent generate Python code using the executor.
    This gives the agent full control over visualization decisions and creation.
    """
    logger.info("=== VISUALIZER AGENT STARTED ===")
    question = state.get("question", "")
    analysis_results = state.get("analysis_results", {})
    
    # Check if we have data to visualize
    if not analysis_results:
        logger.info("❌ VISUALIZER: No analysis results available for visualization")
        result = {
            "visualization_config": {"chart_type": "none", "message": "No data to visualize"},
            "messages": []
        }
        return result
    
    # Handle case where SQL was generated but not executed
    if analysis_results.get("type") == "sql_generated" and "sql_code" in analysis_results:
        sql_code = analysis_results.get("sql_code", "")
        message = analysis_results.get("message", "SQL query generated but could not be executed.")
        logger.info("❌ VISUALIZER: SQL code generated but not executed - missing database connection")
        result = {
            "visualization_config": {
                "chart_type": "code_display", 
                "message": message,
                "sql_code": sql_code,
                "data": []
            },
            "messages": []
        }
        return result
    
    # Extract execution results
    execution_result = analysis_results.get("execution", {})
    result_data = execution_result.get("result", {})
    
    if not isinstance(result_data, dict) or result_data.get("type") != "dataframe":
        logger.info("❌ VISUALIZER: Result data is not a DataFrame, skipping visualization")
        result = {
            "visualization_config": {"chart_type": "none", "message": "Data not suitable for visualization"},
            "messages": []
        }
        return result
    
    # Get the actual data
    data_rows = result_data.get("data", [])
    columns = result_data.get("columns", [])
    
    if not data_rows or not columns:
        logger.info("❌ VISUALIZER: No data available for visualization")
        result = {
            "visualization_config": {"chart_type": "none", "message": "No data available"},
            "messages": []
        }
        return result
    
    try:
        import pandas as pd
        
        # Create DataFrame from the data
        df = pd.DataFrame(data_rows, columns=columns)
        logger.info(f"Created DataFrame with shape {df.shape} and columns: {df.columns.tolist()}")
        
        # Analyze data types for the agent
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Prepare the visualization prompt
        prompt = VISUALIZER_PROMPT.format(
            question=question,
            analysis_results=str(analysis_results)[:500] + "..." if len(str(analysis_results)) > 500 else str(analysis_results),
            result_data=f"DataFrame with {df.shape[0]} rows and {df.shape[1]} columns: {df.columns.tolist()}\nNumeric: {numeric_cols}\nCategorical: {categorical_cols}\nDatetime: {datetime_cols}\nFirst 3 rows:\n{df.head(3).to_string()}"
        )
        
        # Get model to generate visualization code
        messages = [HumanMessage(content=prompt)]
        response = model.invoke(messages)
        content = str(response.content) if hasattr(response, 'content') else str(response)
        
        logger.info(f"Visualizer agent response: {content[:400]}...")
        
        # Extract Python code from the response
        visualization_code = ""
        if "CODE:" in content:
            code_start = content.find("CODE:") + 5
            # Look for code block markers
            if "```python" in content[code_start:]:
                code_block_start = content.find("```python", code_start) + 9
                code_block_end = content.find("```", code_block_start)
                if code_block_end != -1:
                    visualization_code = content[code_block_start:code_block_end].strip()
            elif "```" in content[code_start:]:
                code_block_start = content.find("```", code_start) + 3
                code_block_end = content.find("```", code_block_start)
                if code_block_end != -1:
                    visualization_code = content[code_block_start:code_block_end].strip()
            else:
                # Try to extract code until next section or end
                next_section = content.find("\n\n", code_start)
                if next_section == -1:
                    visualization_code = content[code_start:].strip()
                else:
                    visualization_code = content[code_start:next_section].strip()
        
        if not visualization_code:
            logger.warning("No visualization code found in agent response")
            result = {
                "visualization_config": {"chart_type": "none", "message": "Could not generate visualization code"},
                "messages": [response]
            }
            return result
        
        logger.info(f"Extracted visualization code: {len(visualization_code)} characters")
        logger.debug(f"Visualization code:\n{visualization_code}")
        
        # Prepare the dataframe as CSV string for the python executor
        df_csv = df.to_csv(index=False)
        
        # Modify the code to ensure it returns a Plotly figure and prepare config
        enhanced_code = f"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

# The data is already loaded as df (from CSV)
result_df = df.copy()

{visualization_code}

# Ensure result is a plotly figure and convert to JSON
if 'result' in locals() and hasattr(result, 'to_json'):
    try:
        figure_json = result.to_json()
        
        # Extract title safely from the figure
        figure_title = "Generated Visualization"
        try:
            if hasattr(result.layout, 'title') and hasattr(result.layout.title, 'text'):
                figure_title = result.layout.title.text or "Generated Visualization"
            elif hasattr(result.layout, 'title') and isinstance(result.layout.title, str):
                figure_title = result.layout.title or "Generated Visualization"
        except:
            figure_title = "Generated Visualization"
        
        # Create visualization config
        result = {{
            "chart_type": "plotly",
            "figure_json": figure_json,
            "data": result_df.to_dict('records'),
            "title": figure_title,
            "description": "Agent-generated visualization",
            "columns": result_df.columns.tolist()
        }}
    except Exception as e:
        result = {{
            "chart_type": "none",
            "message": f"Error converting figure to JSON: {{str(e)}}"
        }}
elif 'result' in locals():
    result = {{
        "chart_type": "none",
        "message": f"Result is not a Plotly figure (type: {{type(result).__name__}})"
    }}
else:
    result = {{
        "chart_type": "none",
        "message": "Visualization code did not produce a 'result' variable"
    }}
"""
        
        # Execute the visualization code using the python executor
        execution_result = execute_python_code(enhanced_code, df_csv)
        
        logger.info(f"Python executor result: success={execution_result.get('success')}, error={execution_result.get('error')}")
        logger.debug(f"Full execution result: {execution_result}")
        
        if execution_result.get("success"):
            result_data = execution_result.get("result")
            logger.info(f"Execution succeeded, result type: {type(result_data)}")
            logger.debug(f"Result data: {result_data}")
            
            if isinstance(result_data, dict) and result_data.get("chart_type"):
                visualization_config = result_data
                logger.info(f"✅ VISUALIZER: Successfully generated visualization with chart_type: {visualization_config.get('chart_type')}")
            else:
                logger.warning(f"❌ VISUALIZER: Result is not in expected format: {result_data}")
                visualization_config = {
                    "chart_type": "none",
                    "message": f"Unexpected result format: {type(result_data).__name__}",
                    "code": visualization_code,
                    "raw_result": str(result_data)[:500]
                }
        else:
            error_msg = execution_result.get('error', 'Unknown error')
            output = execution_result.get('output', '')
            logger.error(f"❌ VISUALIZER: Execution failed: {error_msg}")
            if output:
                logger.error(f"❌ VISUALIZER: Execution output: {output}")
            logger.debug(f"❌ VISUALIZER: Generated code that failed:\n{enhanced_code}")
            
            visualization_config = {
                "chart_type": "none",
                "message": f"Visualization execution failed: {error_msg}",
                "code": visualization_code,
                "debug_output": output
            }
        
        # Add execution record
        execution_update = add_agent_execution(
            state, "visualizer",
            {"question": question, "data_shape": df.shape, "columns": df.columns.tolist()},
            {"visualization_config": visualization_config, "generated_code": visualization_code},
            execution_result.get("success", False), 
            "Generated visualization using agent-created Python code"
        )
        
        # Return state update - this will be merged into the main state
        result = {
            "visualization_config": visualization_config,
            "messages": [],
            **execution_update
        }
        
        logger.info(f"✅ VISUALIZER: Returning state update with keys: {list(result.keys())}")
        return result
        
    except Exception as e:
        logger.error(f"Error in visualizer node: {e}")
        # Return fallback
        fallback_config = {
            "chart_type": "none", 
            "message": f"Visualization error: {str(e)}"
        }
        
        logger.info(f"❌ VISUALIZER: Error occurred, returning fallback config: {fallback_config}")
        
        execution_update = add_agent_execution(
            state, "visualizer",
            {"question": question},
            {"error": str(e), "fallback_config": fallback_config},
            False, "Visualization generation failed"
        )
        
        result = {
            "visualization_config": fallback_config,
            "messages": [],
            **execution_update
        }
        
        logger.info(f"❌ VISUALIZER: Returning error state update with keys: {list(result.keys())}")
        return result


# Additional utility functions can be added here as needed