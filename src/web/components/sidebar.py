"""
Enhanced sidebar components for the AI Copilot for Data Teams.
"""
import streamlit as st
import logging
import json
import os
from datetime import datetime
from src.config import EXAMPLE_QUERIES

logger = logging.getLogger("gabi.web.components")

def render_sidebar():
    """
    Render the enhanced sidebar with app information, example queries, and saved figures.
    """
    st.sidebar.markdown("# 🤖 AI Copilot")
    st.sidebar.markdown("---")
    
    # Create tabs in sidebar
    tab1, tab2 = st.sidebar.tabs(["📝 Examples", "💾 Saved Figures"])
    
    with tab1:
        render_example_queries()
    
    with tab2:
        render_saved_figures()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Info")
    
    # Session statistics
    if st.session_state.get("dataframe") is not None:
        df_info = st.session_state.get("df_info", {})
        db_metadata = st.session_state.get("database_metadata", {})
        
        # Show database info if available
        if db_metadata:
            st.sidebar.metric("🗄️ Database", db_metadata.get("db_type", "Unknown"))
            if db_metadata.get("tables"):
                st.sidebar.metric("📋 Tables", len(db_metadata["tables"]))
                with st.sidebar.expander("📋 Table Details", expanded=False):
                    for table_name, table_info in db_metadata["tables"].items():
                        st.write(f"**{table_name}**")
                        st.caption(f"Rows: {table_info.get('rows', 'Unknown')}")
                        st.caption(f"Columns: {table_info.get('columns', 'Unknown')}")
            else:
                # Show current dataset info
                st.sidebar.metric("📊 Dataset", f"{df_info.get('rows', 'N/A')} × {df_info.get('columns', 'N/A')}")
        else:
            # Show CSV file info
            st.sidebar.metric("📁 Dataset", f"{df_info.get('rows', 'N/A')} × {df_info.get('columns', 'N/A')}")
        
        st.sidebar.metric("💬 Queries", len(st.session_state.get("chat_history", [])))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed for the **Confluentia Hackathon**.")
    st.sidebar.markdown("[GitHub Repository](https://github.com/criticic/holonai-hackathon)")
    
    return False  # No longer handling query selection here

def render_example_queries():
    """Render categorized example queries."""
    st.markdown("### 💡 Example Queries")
    
    # Check if we have database metadata for SQL-specific queries
    db_metadata = st.session_state.get("database_metadata", {})
    
    # Use dynamic queries if available, otherwise show guidance message
    if st.session_state.get("categorized_example_queries"):
        st.write("Click on any example to use it:")
        query_categories = st.session_state.categorized_example_queries
        st.markdown("*✨ Generated based on your uploaded data*")
    elif st.session_state.get("dynamic_example_queries"):
        # Fallback for legacy format (list instead of dict)
        st.write("Click on any example to use it:")
        queries_list = st.session_state.dynamic_example_queries
        query_categories = {"General": queries_list}
        st.markdown("*✨ Generated based on your uploaded data*")
    elif db_metadata and db_metadata.get("tables"):
        # Database-specific SQL example queries
        st.write("Click on any example to use it:")
        st.markdown("*�️ Database-specific SQL queries*")
        table_names = list(db_metadata["tables"].keys())
        first_table = table_names[0] if table_names else "your_table"
        
        query_categories = {
            "SQL Analysis": [
                f"SELECT COUNT(*) FROM {first_table} to show total records",
                f"DESCRIBE {first_table} to see the table structure",
                "Show me the schema and relationships between all tables"
            ],
            "Data Exploration": [
                f"SELECT * FROM {first_table} LIMIT 10 to preview the data",
                f"Find columns with missing values in {first_table}",
                "Generate summary statistics for all numeric columns"
            ],
            "Database Insights": [
                "Count records in each table to understand data distribution",
                "Identify primary and foreign key relationships",
                "Find tables with the most recent data updates"
            ]
        }
    else:
        st.info("📁 Upload a CSV file or connect to a database to get personalized example queries!")
        st.write("Here are some general examples:")
        query_categories = EXAMPLE_QUERIES

    for category, queries in query_categories.items():
        with st.expander(category, expanded=True):
            for i, query in enumerate(queries):
                if st.button(query, key=f"sidebar_{category}_{i}", use_container_width=True):
                    # Store the query to be processed
                    st.session_state.sidebar_selected_query = query
                    st.rerun()

def render_saved_figures():
    """Render saved figures management."""
    st.markdown("### 🎨 Saved Visualizations")
    
    # Initialize saved figures if not exists
    if "saved_figures" not in st.session_state:
        st.session_state.saved_figures = []
    
    saved_figures = st.session_state.saved_figures
    
    if not saved_figures:
        st.info("No saved figures yet. Generate visualizations and save them!")
        return
    
    # Clear all figures button
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.saved_figures = []
        st.rerun()
    
    st.markdown("---")
    
    # Display saved figures
    for i, figure_data in enumerate(saved_figures):
        with st.expander(f"📊 {figure_data['title'][:30]}...", expanded=False):
            st.caption(f"Saved: {figure_data['timestamp']}")
            st.caption(f"Query: {figure_data['query'][:50]}...")
            
            # Show thumbnail or description
            if figure_data.get("description"):
                st.text(figure_data["description"])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Load", key=f"load_fig_{i}"):
                    # Load this figure into the main view
                    st.session_state.loaded_figure = figure_data
                    st.rerun()
            
            with col2:
                if st.button("❌ Delete", key=f"del_fig_{i}"):
                    st.session_state.saved_figures.pop(i)
                    st.rerun()

def save_figure(title: str, query: str, viz_config: dict, description: str = ""):
    """Save a figure to the session state."""
    if "saved_figures" not in st.session_state:
        st.session_state.saved_figures = []
    
    figure_data = {
        "id": len(st.session_state.saved_figures),
        "title": title,
        "query": query,
        "viz_config": viz_config,
        "description": description,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.saved_figures.append(figure_data)
    logger.info(f"Saved figure: {title}")
    return True
