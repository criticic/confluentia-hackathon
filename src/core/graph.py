"""
Dynamic LangGraph workflow for the AI Copilot for Data Teams.
This implementation uses a flexible agent orchestration system that can call any agent
at any time in any order as needed to provide comprehensive responses.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import logging
import pandas as pd
from typing import Dict, Any, AsyncIterator, List, Optional, Callable
from langchain_core.runnables import RunnableConfig

from src.core.agents import (
    # State type
    CopilotState,
    # Dynamic agents
    orchestrator_node,
    analyzer_node,
    validator_node,
    planner_node,
    coordinator_node,
    visualizer_node
)

memory = MemorySaver()
logger = logging.getLogger("gabi.core.graph")


def create_copilot_graph():
    """
    Creates and returns the dynamic LangGraph workflow for the AI Data Copilot.
    Uses flexible agent orchestration instead of fixed routing.
    """
    workflow = StateGraph(CopilotState)

    # Add the new dynamic agents
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("visualizer", visualizer_node)

    # Start with the orchestrator
    workflow.add_edge(START, "orchestrator")

    # Dynamic routing function based on orchestrator decisions
    def route_from_orchestrator(state: CopilotState) -> str:
        completion_status = state.get("completion_status", "incomplete")
        next_actions = state.get("next_actions", [])
        
        logger.info(f"Orchestrator routing - Status: {completion_status}, Next: {next_actions}")
        
        if completion_status == "complete":
            # When task is complete, route to coordinator to format final results
            return "coordinator"
        
        if next_actions and len(next_actions) > 0:
            next_agent = next_actions[0]
            
            # Map agent names to node names
            agent_mapping = {
                "analyzer": "analyzer", 
                "validator": "validator",
                "planner": "planner",
                "coordinator": "coordinator",
                "visualizer": "visualizer"
            }
            
            mapped_agent = agent_mapping.get(next_agent, "coordinator")
            logger.info(f"Routing to agent: {mapped_agent}")
            return mapped_agent
        
        # Default to coordinator if no specific next action
        logger.info("No specific next action, routing to coordinator")
        return "coordinator"

    # Set up conditional edges from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "analyzer": "analyzer",
            "validator": "validator", 
            "planner": "planner",
            "coordinator": "coordinator",
            "visualizer": "visualizer",
            END: END,
        }
    )

    # All other agents route back to orchestrator for next decisions
    workflow.add_edge("analyzer", "orchestrator")
    workflow.add_edge("validator", "orchestrator") 
    workflow.add_edge("planner", "orchestrator")
    workflow.add_edge("visualizer", "orchestrator")
    
    # Coordinator decides whether to continue or end
    def route_from_coordinator(state: CopilotState) -> str:
        completion_status = state.get("completion_status", "incomplete")
        coordinator_decision = state.get("coordinator_decision", "")
        
        logger.info(f"Coordinator routing - Status: {completion_status}")
        
        if completion_status == "complete" or "DECISION: COMPLETE" in coordinator_decision:
            return END
        else:
            return "orchestrator"
    
    workflow.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "orchestrator": "orchestrator",
            END: END,
        }
    )

    graph = workflow.compile(checkpointer=memory)
    logger.info("Dynamic AI Copilot graph compiled successfully")
    
    return graph


class StreamEvent(Dict):
    """Event type for streaming copilot updates."""
    type: str
    data: Any
    node: Optional[str]

async def stream_copilot_query(
    question: str,
    dataframe: pd.DataFrame,
    stream_handler: Optional[Callable[[StreamEvent], None]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    database_metadata: Optional[Dict[str, Any]] = None,
    data_source_type: str = "csv",
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream the execution of the copilot pipeline, yielding intermediate results.
    """
    thread_id = "copilot-stream-" + str(hash(question))[:8]
    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Creating streaming copilot graph for query: '{question[:50]}...'")
    graph = create_copilot_graph()

    # Prepare initial state from the DataFrame
    # Use more robust CSV conversion with proper escaping
    try:
        df_head_str = dataframe.to_csv(index=False, quotechar='"', escapechar='\\')
    except Exception as e:
        logger.warning(f"Error converting DataFrame to CSV: {e}. Using fallback method.")
        # Fallback: use string representation if CSV conversion fails
        df_head_str = str(dataframe.head(10))
    
    column_info = "\n".join([f"- {col} ({dtype})" for col, dtype in dataframe.dtypes.items()])

    initial_state: CopilotState = {
        # Core data
        "question": question,
        "original_question": question,
        "dataframe_str": df_head_str,
        "column_info": column_info,
        
        # Agent execution tracking
        "agent_executions": [],
        "current_reasoning": "",
        "next_actions": [],
        "completion_status": "incomplete",
        
        # Agent-specific outputs
        "research_findings": "",
        "analysis_results": {},
        "validation_results": "",
        "planning_output": "",
        "coordinator_decision": "",
        "visualization_config": {},  # Initialize visualization config
        
        # Database context
        "database_metadata": database_metadata,
        "data_source_type": data_source_type,
    }

    logger.info("Beginning streaming copilot workflow execution")
    event_count = 0
    async for event in graph.astream(initial_state, thread_config, stream_mode="updates"):
        event_count += 1
        node_name = list(event.keys())[0] if event else None
        event_data = {
            "type": "update",
            "data": event,
            "node": node_name,
        }

        if stream_handler:
            logger.debug(f"Streaming event from {node_name} node")
            stream_event = StreamEvent(event_data)
            stream_handler(stream_event)
        
        yield event_data
    
    logger.info(f"Streaming workflow completed with {event_count} events processed")
