"""
Clean state management utilities for the new AI Copilot web interface.
Simple, robust state handling without complex placeholders.
"""
import streamlit as st
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger("gabi.web.state")

def initialize_session_state() -> None:
    """Initialize all required session state variables with safe defaults."""
    defaults = {
        "initialized": True,
        "dataframe": None,
        "current_file_id": "",
        "dynamic_example_queries": [],
        "categorized_example_queries": {},
        "chat_history": [],
        "is_processing": False,
        "current_analysis": None,
        "analysis_complete": False,
        "last_query": "",
        "query_count": 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_analysis_state() -> None:
    """Reset analysis-related state variables."""
    st.session_state.current_analysis = None
    st.session_state.analysis_complete = False
    st.session_state.is_processing = False

def add_to_chat_history(question: str, results: Dict[str, Any]) -> None:
    """Add a new chat interaction to history."""
    chat_entry = {
        "question": question,
        "results": results,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "query_id": st.session_state.get("query_count", 0)
    }
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append(chat_entry)
    st.session_state.query_count = st.session_state.get("query_count", 0) + 1
    st.session_state.last_query = question

def clear_chat_history() -> None:
    """Clear all chat history."""
    st.session_state.chat_history = []
    st.session_state.query_count = 0
    st.session_state.last_query = ""

def get_current_dataframe():
    """Get the current dataframe safely."""
    return st.session_state.get("dataframe")

def set_dataframe(df, file_id: str) -> None:
    """Set the current dataframe and update related state."""
    st.session_state.dataframe = df
    st.session_state.current_file_id = file_id
    # Reset analysis state when new data is loaded
    reset_analysis_state()

def is_data_loaded() -> bool:
    """Check if data is currently loaded."""
    return st.session_state.get("dataframe") is not None

def is_processing() -> bool:
    """Check if a query is currently being processed."""
    return st.session_state.get("is_processing", False)

def set_processing(processing: bool) -> None:
    """Set the processing state."""
    st.session_state.is_processing = processing

def get_chat_history() -> List[Dict[str, Any]]:
    """Get the current chat history."""
    return st.session_state.get("chat_history", [])

def get_example_queries() -> List[str]:
    """Get the current example queries."""
    return st.session_state.get("dynamic_example_queries", [])

def set_example_queries(queries: List[str]) -> None:
    """Set the example queries."""
    st.session_state.dynamic_example_queries = queries

def get_session_info() -> Dict[str, Any]:
    """Get summary information about the current session."""
    return {
        "data_loaded": is_data_loaded(),
        "is_processing": is_processing(),
        "chat_count": len(get_chat_history()),
        "last_query": st.session_state.get("last_query", ""),
        "query_count": st.session_state.get("query_count", 0)
    }

class SessionManager:
    """Context manager for handling session state safely."""
    
    def __init__(self):
        self.initial_state = {}
    
    def __enter__(self):
        # Save current state
        self.initial_state = {
            key: st.session_state.get(key) 
            for key in ["is_processing", "current_analysis", "analysis_complete"]
        }
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore state if there was an exception
        if exc_type is not None:
            for key, value in self.initial_state.items():
                if value is not None:
                    st.session_state[key] = value
            logger.error(f"Session manager caught exception: {exc_type.__name__}: {exc_val}")

def safe_session_update(updates: Dict[str, Any]) -> bool:
    """Safely update multiple session state variables."""
    try:
        for key, value in updates.items():
            st.session_state[key] = value
        return True
    except Exception as e:
        logger.error(f"Failed to update session state: {e}")
        return False