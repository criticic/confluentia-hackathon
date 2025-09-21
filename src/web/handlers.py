"""
Clean event handlers for the new AI Copilot web interface.
Simple, robust handling without complex placeholders.
"""
import streamlit as st
import pandas as pd
import logging
import asyncio
from typing import Dict, Any, List

from src.core.graph import stream_copilot_query
from src.core.agents import generate_dynamic_example_queries, generate_database_specific_queries
from src.web.state import (
    set_processing, add_to_chat_history, get_current_dataframe,
    set_dataframe, get_chat_history
)

logger = logging.getLogger("gabi.web.handlers")

def handle_file_upload(uploaded_file_or_df) -> bool:
    """
    Handle CSV file upload or DataFrame processing.
    Returns True if successful, False otherwise.
    """
    try:
        if uploaded_file_or_df is None:
            return False
        
        # Handle DataFrame input (from database connections)
        if isinstance(uploaded_file_or_df, pd.DataFrame):
            df = uploaded_file_or_df
            file_id = f"database_query_{hash(str(df.columns.tolist()) + str(df.shape))}"
            file_name = "Database Query Result"
        else:
            # Handle file upload
            uploaded_file = uploaded_file_or_df
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            file_name = uploaded_file.name
            
            # Check if this is a new file
            if file_id == st.session_state.get("current_file_id", ""):
                return True  # File already loaded
            
            df = pd.read_csv(uploaded_file)
        
        with st.spinner("Loading data..."):
            # Update state
            set_dataframe(df, file_id)
            
            # Generate AI-powered example queries
            try:
                # Check if this is database metadata (overview) vs actual data
                db_metadata = st.session_state.get("database_metadata", {})
                
                if db_metadata and db_metadata.get("tables") and len(df.columns) <= 5 and 'Table' in df.columns:
                    # This is a database overview, generate queries based on actual table schemas
                    dynamic_queries = generate_database_specific_queries(db_metadata)
                else:
                    # This is regular data, use standard AI generation
                    dynamic_queries = generate_dynamic_example_queries(df)
                
                # Store categorized queries
                st.session_state.categorized_example_queries = dynamic_queries
                
                # Convert to flat list for backward compatibility
                all_queries = []
                for category, queries in dynamic_queries.items():
                    all_queries.extend(queries)
                st.session_state.dynamic_example_queries = all_queries
                
                logger.info(f"Generated {len(all_queries)} dynamic example queries using AI across {len(dynamic_queries)} categories")
            except Exception as e:
                logger.error(f"Error generating dynamic queries: {e}")
                # Fallback to basic queries
                st.session_state.dynamic_example_queries = [
                    "Show me a summary of the data",
                    "What are the data types of each column?",
                    "Are there any missing values?",
                    "Show me the first 10 rows"
                ]
                st.session_state.categorized_example_queries = {"General": st.session_state.dynamic_example_queries}
                logger.info("Using fallback query generation")
            
            # Clear chat history for new data
            st.session_state.chat_history = []
        
        st.success("✅ Data loaded successfully!")
        logger.info(f"Loaded data: {file_name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return True
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return False

async def process_user_query(question: str) -> Dict[str, Any]:
    """
    Process a user query and return results.
    Clean async processing without complex state management.
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "No data loaded"}
        
        set_processing(True)
        
        # Create status containers
        status_container = st.empty()
        progress_container = st.empty()
        
        # Initialize result storage
        final_state = {}
        event_count = 0
        
        # Process the query
        status_container.info("🤖 Starting analysis...")
        logger.info(f"Processing query: {question}")
        
        async for event in stream_copilot_query(
            question=question,
            dataframe=df,
            stream_handler=None,
            chat_history=get_chat_history(),
            database_metadata=st.session_state.get("database_metadata", {}),
            data_source_type=st.session_state.get("data_source_type", "csv")
        ):
            event_count += 1
            final_state = event.get("data", {})
            
            # Update progress
            progress_container.info(f"🔄 Processing... (step {event_count})")
        
        # Clear status containers
        status_container.empty()
        progress_container.empty()
        
        # Extract results
        results = extract_analysis_results(final_state)
        
        logger.info(f"Query processed successfully with {event_count} steps")
        return results
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        return {"error": str(e)}
    
    finally:
        set_processing(False)

def extract_analysis_results(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract analysis results from the final workflow state.
    Clean extraction logic without complex parsing.
    """
    try:
        results = {}
        
        logger.info(f"=== EXTRACTING RESULTS ===")
        logger.info(f"Final state keys: {list(final_state.keys())}")
        
        # Check if visualization_config exists at the top level of final_state
        if "visualization_config" in final_state:
            results["viz_config"] = final_state["visualization_config"]
            logger.info(f"✅ Found top-level visualization_config: {final_state['visualization_config']}")
        
        for node, node_data in final_state.items():
            if not isinstance(node_data, dict):
                continue
            
            logger.info(f"Processing node '{node}' with keys: {list(node_data.keys())}")
            
            if node == "analyzer":
                # Extract execution output
                execution_result = node_data.get("execution_result", {})
                analysis_results = node_data.get("analysis_results", {})
                
                if execution_result and execution_result.get("success"):
                    output = execution_result.get("output", "")
                    if output:
                        results["execution_output"] = output
                        logger.info(f"Found execution output: {len(output)} characters")
                
                elif analysis_results:
                    execution_data = analysis_results.get("execution", {})
                    if execution_data and execution_data.get("output"):
                        results["execution_output"] = execution_data["output"]
                        logger.info("Found analysis output in analysis_results")
                
                # Extract Python code
                if node_data.get("python_code"):
                    results["python_code"] = node_data["python_code"]
                    logger.info("Found Python code")
                
                # Extract visualization config from analyzer
                if node_data.get("visualization_config"):
                    results["viz_config"] = node_data["visualization_config"]
                    logger.info(f"✅ Found visualization config from analyzer: {node_data['visualization_config']}")
                    logger.info(f"Analyzer viz config chart_type: {node_data['visualization_config'].get('chart_type')}")
                    logger.info(f"Analyzer viz config has data: {bool(node_data['visualization_config'].get('data'))}")
                    logger.info(f"Analyzer viz config has figure_json: {bool(node_data['visualization_config'].get('figure_json'))}")
                else:
                    logger.info("No visualization_config found in analyzer node_data")
            
            elif node == "coordinator":
                # Extract coordinator decision and synthesis
                if node_data.get("coordinator_decision"):
                    coordinator_decision = node_data["coordinator_decision"]
                    
                    # Parse the response to extract synthesis and reasoning if it's a string
                    if isinstance(coordinator_decision, str):
                        # Extract SYNTHESIS and REASONING from the structured response
                        synthesis_start = coordinator_decision.find("SYNTHESIS:")
                        reasoning_start = coordinator_decision.find("REASONING:")
                        
                        synthesis = ""
                        reasoning = ""
                        
                        if synthesis_start != -1:
                            if reasoning_start != -1 and reasoning_start > synthesis_start:
                                synthesis = coordinator_decision[synthesis_start + 10:reasoning_start].strip()
                                reasoning = coordinator_decision[reasoning_start + 10:].strip()
                            else:
                                synthesis = coordinator_decision[synthesis_start + 10:].strip()
                        
                        if synthesis:  # Only create structured format if we found synthesis
                            results["coordinator_decision"] = {
                                "synthesis": synthesis,
                                "reasoning": reasoning,
                                "full_response": coordinator_decision
                            }
                            logger.info(f"Parsed coordinator synthesis: {len(synthesis)} chars")
                        else:
                            results["coordinator_decision"] = coordinator_decision
                            logger.info("Using coordinator decision as-is (no SYNTHESIS found)")
                    else:
                        results["coordinator_decision"] = coordinator_decision
                        logger.info("Found coordinator decision")
                
                # Extract visualization config from coordinator
                if node_data.get("visualization_config"):
                    results["viz_config"] = node_data["visualization_config"]
                    logger.info(f"✅ Found visualization config from coordinator: {node_data['visualization_config']}")
                    logger.info(f"Coordinator viz config chart_type: {node_data['visualization_config'].get('chart_type')}")
                    logger.info(f"Coordinator viz config has data: {bool(node_data['visualization_config'].get('data'))}")
                    logger.info(f"Coordinator viz config has figure_json: {bool(node_data['visualization_config'].get('figure_json'))}")
                else:
                    logger.info("No visualization_config found in coordinator node_data")
            
            elif node == "visualizer":
                # Extract visualization config from visualizer node
                if node_data.get("visualization_config"):
                    results["viz_config"] = node_data["visualization_config"]
                    logger.info(f"✅ Found visualization config from visualizer: {node_data['visualization_config']}")
                    logger.info(f"Visualizer chart_type: {node_data['visualization_config'].get('chart_type')}")
                    logger.info(f"Visualizer has data: {bool(node_data['visualization_config'].get('data'))}")
                    logger.info(f"Visualizer has figure_json: {bool(node_data['visualization_config'].get('figure_json'))}")
                else:
                    logger.info("❌ No visualization_config found in visualizer node")
            
            # Extract agent executions from any node that has them
            if node_data.get("agent_executions"):
                results["agent_executions"] = node_data["agent_executions"]
                logger.info(f"Found {len(node_data['agent_executions'])} agent executions")
        
        logger.info(f"=== FINAL RESULTS ===")
        logger.info(f"Extracted results with keys: {list(results.keys())}")
        if "viz_config" in results:
            logger.info(f"✅ Final viz_config chart_type: {results['viz_config'].get('chart_type')}")
        else:
            logger.info("❌ No viz_config in final results")
        
        return results
    
    except Exception as e:
        logger.error(f"Error extracting results: {e}")
        return {"error": f"Error extracting results: {e}"}

def handle_query_submission(question: str) -> bool:
    """
    Handle the submission of a user query.
    Returns True if handled successfully, False otherwise.
    """
    try:
        if not question or not question.strip():
            return False
        
        if get_current_dataframe() is None:
            st.warning("Please upload a CSV file first.")
            return False
        
        if st.session_state.get("is_processing", False):
            st.info("🔄 Processing previous request...")
            return False
        
        # Show processing indicator
        with st.spinner("🔄 Processing your query..."):
            # Run the async query processing
            results = asyncio.run(process_user_query(question))
            
            if results.get("error"):
                st.error(f"❌ Error: {results['error']}")
                return False
            else:
                # Add to chat history (this will be displayed by the chat interface)
                st.session_state.query_count = st.session_state.get("query_count", 0) + 1
                add_to_chat_history(question, results)
                
                st.success("✅ Analysis Complete!")
                
                # Force rerun to update the chat interface with new results
                st.rerun()
                
                return True
    
    except Exception as e:
        logger.error(f"Error handling query submission: {e}")
        st.error(f"Error processing query: {e}")
        return False