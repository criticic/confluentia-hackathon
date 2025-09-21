"""
Clean agent execution tracker for the new AI Copilot web interface.
Simple display of agent execution steps without complex state management.
"""
import streamlit as st
import logging
from typing import Dict, Any, List

logger = logging.getLogger("gabi.web.agent_tracker")

def get_agent_icon(agent_name: str) -> str:
    """Get icon for agent type."""
    icons = {
        "orchestrator": "🎯",
        "analyzer": "🔍", 
        "validator": "✅",
        "planner": "📋",
        "coordinator": "🤝"
    }
    return icons.get(agent_name.lower(), "🤖")

def display_agent_execution(execution: Dict[str, Any], index: int) -> None:
    """Display a single agent execution step with detailed information."""
    try:
        agent_name = execution.get("agent_name", "Unknown")
        success = execution.get("success", False)
        reasoning = execution.get("reasoning", "")
        timestamp = execution.get("timestamp", "")
        input_data = execution.get("input_data", {})
        output_data = execution.get("output_data", {})
        
        # Get status and icon
        status_icon = "✅" if success else "❌"
        agent_icon = get_agent_icon(agent_name)
        
        # Create an expander for each agent step
        with st.expander(f"{index}. {agent_icon} {status_icon} **{agent_name.title()}**: {reasoning}", expanded=False):
            
            # Show timestamp
            if timestamp:
                st.caption(f"⏰ {timestamp}")
            
            # Show inputs if available
            if input_data:
                st.markdown("**📥 Inputs:**")
                for key, value in input_data.items():
                    if key == "question" and value:
                        st.write(f"• **Question**: {value}")
                    elif key == "dataframe_info" and value:
                        st.write(f"• **Data**: {value}")
                    elif value and len(str(value)) < 200:
                        st.write(f"• **{key.title()}**: {value}")
            
            # Show outputs with detailed information
            if output_data:
                st.markdown("**📤 Outputs:**")
                
                # For analyzer: look for analysis_results and execution data
                if agent_name.lower() == "analyzer" and isinstance(output_data, dict):
                    # Check for analysis results
                    if "content" in output_data:
                        analysis_content = output_data["content"]
                        st.markdown("**� Analysis Response:**")
                        
                        # Extract and display code if present
                        if "CODE:" in analysis_content:
                            code_start = analysis_content.find("CODE:") + 5
                            next_markers = ["ANALYSIS:", "RESULT:", "APPROACH:", "\n\n#", "\n---"]
                            code_end = len(analysis_content)
                            for marker in next_markers:
                                marker_pos = analysis_content.find(marker, code_start)
                                if marker_pos != -1:
                                    code_end = min(code_end, marker_pos)
                            
                            code_section = analysis_content[code_start:code_end].strip()
                            if code_section.startswith("```python"):
                                code_section = code_section[9:]
                            if code_section.startswith("```"):
                                code_section = code_section[3:]
                            if code_section.endswith("```"):
                                code_section = code_section[:-3]
                            
                            if code_section.strip():
                                st.markdown("**🐍 Generated Python Code:**")
                                st.code(code_section.strip(), language="python")
                        
                        # Display approach and analysis parts
                        if "APPROACH:" in analysis_content:
                            approach_start = analysis_content.find("APPROACH:") + 9
                            approach_end = analysis_content.find("CODE:", approach_start)
                            if approach_end == -1:
                                approach_end = analysis_content.find("ANALYSIS:", approach_start)
                            if approach_end == -1:
                                approach_end = len(analysis_content)
                            
                            approach = analysis_content[approach_start:approach_end].strip()
                            if approach:
                                st.markdown("**� Approach:**")
                                st.write(approach)
                        
                        if "ANALYSIS:" in analysis_content:
                            analysis_start = analysis_content.find("ANALYSIS:") + 9
                            analysis = analysis_content[analysis_start:].strip()
                            if analysis:
                                st.markdown("**🔍 Analysis:**")
                                st.write(analysis)
                    
                    # Check for execution results - PRIORITIZE THESE OVER LLM RESPONSE
                    if "execution" in output_data:
                        execution_result = output_data["execution"]
                        if isinstance(execution_result, dict):
                            if execution_result.get("success"):
                                st.markdown("**✅ Execution Results:**")
                                st.success("✅ **Real data from database execution**")
                                
                                if execution_result.get("output"):
                                    st.code(execution_result["output"])
                                if execution_result.get("result"):
                                    result = execution_result["result"]
                                    if isinstance(result, dict) and result.get("data"):
                                        st.write(f"**Result Type**: {result.get('type', 'Unknown')}")
                                        
                                        # Emphasize this is real data
                                        st.markdown("**📊 Actual Database Results:**")
                                        if len(str(result["data"])) < 1000:
                                            st.json(result["data"])
                                        else:
                                            st.write(f"**Data**: [Large dataset with {len(str(result['data']))} characters]")
                                            # Show preview of large datasets
                                            if isinstance(result["data"], list) and len(result["data"]) > 0:
                                                st.write("**Preview (first 5 rows):**")
                                                st.json(result["data"][:5])
                                        
                                        if result.get("shape"):
                                            st.write(f"**Shape**: {result['shape'][0]} rows × {result['shape'][1]} columns")
                            else:
                                st.markdown("**❌ Execution Failed:**")
                                if execution_result.get("error"):
                                    st.error(execution_result["error"])
                
                # For coordinator: look for decision
                elif agent_name.lower() == "coordinator" and isinstance(output_data, dict):
                    if "decision" in output_data:
                        decision = output_data["decision"]
                        st.markdown("**🤝 Coordinator Decision:**")
                        
                        if isinstance(decision, str):
                            # Parse SYNTHESIS and REASONING
                            if "SYNTHESIS:" in decision:
                                synthesis_start = decision.find("SYNTHESIS:")
                                reasoning_start = decision.find("REASONING:")
                                
                                if synthesis_start != -1:
                                    if reasoning_start != -1 and reasoning_start > synthesis_start:
                                        synthesis = decision[synthesis_start + 10:reasoning_start].strip()
                                        reasoning_text = decision[reasoning_start + 10:].strip()
                                        st.markdown(f"**Summary:** {synthesis}")
                                        if reasoning_text:
                                            st.markdown(f"**Reasoning:** {reasoning_text}")
                                    else:
                                        synthesis = decision[synthesis_start + 10:].strip()
                                        st.markdown(f"**Summary:** {synthesis}")
                                else:
                                    st.write(decision)
                            else:
                                st.write(decision)
                        else:
                            st.write(str(decision))
                
                # For other agents: show all output data
                else:
                    for key, value in output_data.items():
                        if value and len(str(value)) < 500:
                            st.write(f"• **{key.title()}**: {value}")
                        elif value:
                            st.write(f"• **{key.title()}**: [Large data - {len(str(value))} characters]")
    
    except Exception as e:
        logger.error(f"Error displaying agent execution: {e}")
        st.error(f"Error displaying step {index}: {e}")

def render_agent_tracker(agent_executions: List[Dict[str, Any]]) -> None:
    """
    Render agent execution tracker with detailed expandable sections.
    """
    try:
        if not agent_executions:
            st.info("No agent execution data available")
            return
        
        # Display each execution in an expandable format (no extra header since it's in an expander)
        for i, execution in enumerate(agent_executions, 1):
            display_agent_execution(execution, i)
        
        # Summary statistics
        total_executions = len(agent_executions)
        successful_executions = sum(1 for ex in agent_executions if ex.get("success", False))
        
        st.divider()
        st.caption(f"📊 Total: {total_executions} steps | ✅ Successful: {successful_executions}")
    
    except Exception as e:
        logger.error(f"Error rendering agent tracker: {e}")
        st.error("Error displaying agent execution tracker")

def display_execution_summary(agent_executions: List[Dict[str, Any]]) -> None:
    """
    Display a simple summary of agent executions.
    """
    try:
        if not agent_executions:
            return
        
        # Count agents
        agent_counts = {}
        for execution in agent_executions:
            agent_name = execution.get("agent_name", "Unknown")
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
        
        # Display summary
        st.markdown("#### 📈 Execution Summary")
        for agent, count in agent_counts.items():
            icon = get_agent_icon(agent)
            st.write(f"{icon} **{agent.title()}**: {count} time{'s' if count != 1 else ''}")
    
    except Exception as e:
        logger.error(f"Error displaying execution summary: {e}")

def render_simple_agent_log(agent_executions: List[Dict[str, Any]]) -> None:
    """
    Render a very simple agent execution log for compact display.
    """
    try:
        if not agent_executions:
            return
        
        for i, execution in enumerate(agent_executions, 1):
            agent_name = execution.get("agent_name", "Unknown")
            success = execution.get("success", False)
            
            status_icon = "✅" if success else "❌"
            agent_icon = get_agent_icon(agent_name)
            
            st.write(f"{i}. {agent_icon} {status_icon} **{agent_name.title()}**")
    
    except Exception as e:
        logger.error(f"Error rendering simple agent log: {e}")
        st.error("Error displaying agent log")