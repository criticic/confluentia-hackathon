"""
Main web interface for the AI Copilot for Data Teams.
"""
import streamlit as st
import pandas as pd
import logging
import asyncio
import json
from typing import Dict, Any

# Configure logging and Streamlit page settings first
from src.utils.logging import configure_logging
configure_logging()

st.set_page_config(page_title="AI Copilot for Data Teams", layout="wide")

# Import project modules
from src.web.state import initialize_session_state, get_current_dataframe, get_session_info, add_to_chat_history
from src.web.handlers import handle_file_upload, handle_query_submission
from src.web.components.sidebar import render_sidebar
from src.web.components.visualization import render_visualization, auto_visualize_dataframe, display_data_table
from src.utils.db_connector import (
    connect_and_query_sql, 
    connect_and_query_mongo, 
    connect_and_analyze_database,
    analyze_mongo_database,
    get_database_schema
)
from src.core.memory import initialize_memory, save_user_logic
from src.tools.python_executor import execute_python_code
from streamlit_local_storage import LocalStorage

logger = logging.getLogger("gabi.web.app")
localS = LocalStorage()

# --- UI Rendering Functions ---

def render_data_loader():
    """Renders the UI for loading data from CSV or a database."""
    st.subheader("1. Load Your Data")
    
    tab1, tab2 = st.tabs(["📁 Upload CSV", "🗄️ Connect to Database"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type="csv", help="Upload a CSV file to begin your analysis."
        )
        if uploaded_file:
            handle_file_upload(uploaded_file)

    with tab2:
        render_database_connectors()

def render_database_connectors():
    """Renders the UI for connecting to various databases with automatic analysis."""
    db_type = st.selectbox("Select Database Type", ["PostgreSQL", "MySQL", "SQLite", "MongoDB"])

    with st.form(key=f"{db_type}_connect_form"):
        st.write(f"**{db_type} Connection Details**")
        
        # Initialize variables
        conn = {}
        query = ""
        collection = ""
        analysis_mode = ""
        
        if db_type in ["PostgreSQL", "MySQL"]:
            conn = {
                "host": st.text_input("Host", "localhost"),
                "port": st.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306),
                "user": st.text_input("User", "root"),
                "password": st.text_input("Password", type="password"),
                "database": st.text_input("Database Name"),
            }
            
            analysis_mode = st.radio(
                "Analysis Mode", 
                ["Analyze All Tables", "Custom SQL Query"],
                help="Choose to automatically analyze all tables or run a specific query"
            )
            
            if analysis_mode == "Custom SQL Query":
                query = st.text_area("SQL Query", "SELECT * FROM your_table LIMIT 1000;")
        
        elif db_type == "SQLite":
            conn = {"database": st.text_input("Database File Path", "data.db")}
            
            analysis_mode = st.radio(
                "Analysis Mode", 
                ["Analyze All Tables", "Custom SQL Query"],
                help="Choose to automatically analyze all tables or run a specific query"
            )
            
            if analysis_mode == "Custom SQL Query":
                query = st.text_area("SQL Query", "SELECT * FROM your_table LIMIT 1000;")
        
        elif db_type == "MongoDB":
            conn = {
                "uri": st.text_input("MongoDB Connection URI", "mongodb://localhost:27017/"),
                "database": st.text_input("Database Name"),
            }
            
            analysis_mode = st.radio(
                "Analysis Mode", 
                ["Analyze All Collections", "Query Specific Collection"],
                help="Choose to automatically analyze all collections or query a specific one"
            )
            
            if analysis_mode == "Query Specific Collection":
                collection = st.text_input("Collection Name")
                query = st.text_area("Filter (JSON)", "{}")

        submitted = st.form_submit_button("🔗 Connect and Analyze Database")
        
        if submitted:
            if db_type == "MongoDB":
                if analysis_mode == "Analyze All Collections":
                    # Analyze all collections
                    overview_df, metadata = analyze_mongo_database(conn)
                    if not overview_df.empty:
                        st.session_state.database_metadata = metadata
                        st.session_state.data_source_type = "mongodb"
                        st.session_state.db_connection_details = conn
                        st.session_state.db_type = db_type
                        handle_file_upload(overview_df)
                        st.rerun()  # Force sidebar to update with database-specific queries
                else:
                    # Query specific collection
                    result_df = connect_and_query_mongo(conn, collection, query)
                    if result_df is not None:
                        st.session_state.db_connection_details = conn
                        st.session_state.db_type = db_type
                        st.session_state.selected_collection = collection
                        handle_file_upload(result_df)
            else:
                if analysis_mode == "Analyze All Tables":
                    # Analyze all tables in the database
                    overview_df, metadata = connect_and_analyze_database(db_type.lower(), conn)
                    if not overview_df.empty:
                        st.session_state.database_metadata = metadata
                        st.session_state.data_source_type = "database"
                        st.session_state.db_connection_details = conn
                        st.session_state.db_type = db_type
                        handle_file_upload(overview_df)
                        st.rerun()  # Force sidebar to update with database-specific queries
                else:
                    # Run custom query
                    result_df = connect_and_query_sql(db_type.lower(), conn, query)
                    if result_df is not None:
                        st.session_state.db_connection_details = conn
                        st.session_state.db_type = db_type
                        handle_file_upload(result_df)

def render_chat_interface():
    """Renders the improved chat interface with better layout."""
    st.subheader("2. 💬 Analysis & Results")
    
    # Check for pending query from sidebar
    if st.session_state.get("sidebar_selected_query"):
        query = st.session_state.sidebar_selected_query
        st.session_state.sidebar_selected_query = None  # Clear it
        handle_query_submission(query)
    
    # Chat history container - show conversation history first
    chat_container = st.container()
    
    with chat_container:
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.markdown("### 📜 Conversation History")
            
            # Show recent analyses in chronological order (oldest first)
            recent_history = st.session_state.chat_history[-5:]  # Show last 5
            for i, message in enumerate(recent_history):
                query_id = len(st.session_state.chat_history) - len(recent_history) + i + 1
                
                # Expand the most recent query (last one in the list)
                is_expanded = (i == len(recent_history) - 1)
                
                with st.expander(f"🔍 Query {query_id}: {message['question'][:60]}...", expanded=is_expanded):
                    display_analysis_results(message["results"], query_id)
                
                st.markdown("---")
    
    # Input section at the bottom after all history
    st.markdown("### 💭 Ask a Question")
    
    # Main input at bottom
    prompt = st.chat_input("Ask a question about your data...")
    if prompt:
        handle_query_submission(prompt)


def display_analysis_results(results: Dict[str, Any], query_id: int):
    """
    Displays the analysis results in the improved UI layout:
    1. Agent execution steps (detailed, at top)
    2. Coordinator synthesis (summary)
    3. Visualization with save option
    4. Raw data/code (collapsed)
    """
    if not results or results.get("error"):
        st.error(results.get("error", "An unknown error occurred."))
        return

    # 1. AGENT EXECUTION STEPS (TOP - DETAILED)
    if "agent_executions" in results:
        st.markdown("### 🤖 Agent Execution Steps")
        with st.expander("📋 Detailed Agent Activity", expanded=True):
            from src.web.components.agent_tracker import render_agent_tracker
            render_agent_tracker(results["agent_executions"])

    # 2. COORDINATOR SYNTHESIS (SUMMARY)
    if "coordinator_decision" in results:
        st.markdown("### 📊 Analysis Summary")
        decision = results["coordinator_decision"]
        if isinstance(decision, dict) and "synthesis" in decision:
            st.markdown(decision["synthesis"])
        elif isinstance(decision, str):
            synthesis_part = decision.split("SYNTHESIS:")[-1].split("REASONING:")[0].strip()
            if synthesis_part:
                st.markdown(synthesis_part)
            else:
                st.markdown(decision)

    # 3. VISUALIZATION WITH SAVE OPTION
    viz_displayed = False
    if "viz_config" in results and results["viz_config"]:
        viz_config = results["viz_config"]
        chart_type = viz_config.get("chart_type", "none")
        
        logger.info(f"=== VISUALIZATION SECTION ===")
        logger.info(f"Found viz_config with chart_type: {chart_type}")
        logger.info(f"Has figure_json: {bool(viz_config.get('figure_json'))}")
        logger.info(f"Has data: {bool(viz_config.get('data'))}")
        
        # Show visualization for any chart type (including code_display)
        if chart_type != "none":
            st.markdown("### 📈 Visualization")
            
            # Debug logging for visualization config
            logger.info(f"About to render visualization with config: {viz_config}")
            
            # Create columns for visualization and save button
            viz_col, save_col = st.columns([4, 1])
            
            with viz_col:
                viz_displayed = render_visualization(viz_config, viz_config.get("data", []))
            
            with save_col:
                # Save figure button (only for plotly charts)
                if chart_type == "plotly" and st.button("💾 Save Figure", key=f"save_viz_{query_id}"):
                    from src.web.components.sidebar import save_figure
                    query = st.session_state.get("chat_history", [{}])[-1].get("question", "Unknown Query")
                    title = st.text_input("Figure Title:", value=f"Analysis {query_id}", key=f"fig_title_{query_id}")
                    if title:
                        save_figure(
                            title=title,
                            query=query,
                            viz_config=viz_config,
                            description=f"Generated from: {query[:100]}..."
                        )
                        st.success("📊 Figure saved!")
        else:
            logger.info("Chart type is 'none', skipping visualization display")

    # 4. DATA RESULTS (IF AVAILABLE)
    if "execution_output" in results:
        output = results["execution_output"]
        if isinstance(output, dict) and output.get("type") == "dataframe":
            st.markdown("### 📋 Data Results")
            df = pd.DataFrame(output["data"])
            st.dataframe(df, use_container_width=True)
            
            # Auto-visualize if no explicit visualization was created
            if not viz_displayed and not df.empty:
                st.markdown("### 📊 Auto-Generated Visualization")
                auto_viz_col, auto_save_col = st.columns([4, 1])
                with auto_viz_col:
                    auto_visualize_dataframe(df)
                with auto_save_col:
                    if st.button("💾 Save Auto-Viz", key=f"save_auto_viz_{query_id}"):
                        from src.web.components.sidebar import save_figure
                        query = st.session_state.get("chat_history", [{}])[-1].get("question", "Unknown Query")
                        title = st.text_input("Auto-Viz Title:", value=f"Auto Analysis {query_id}", key=f"auto_fig_title_{query_id}")
                        if title:
                            # Create a basic viz config for the auto-generated chart
                            auto_viz_config = {
                                "chart_type": "auto",
                                "data": df.to_dict('records'),
                                "title": title
                            }
                            save_figure(
                                title=title,
                                query=query,
                                viz_config=auto_viz_config,
                                description=f"Auto-generated from: {query[:100]}..."
                            )
                            st.success("📊 Auto-visualization saved!")
        elif isinstance(output, str) and output:
            with st.expander("📋 Execution Output", expanded=False):
                st.code(output)
    
    # 5. TECHNICAL DETAILS (COLLAPSED BY DEFAULT)
    with st.expander("🔬 Technical Details", expanded=False):
        # Show Python code if available
        if "python_code" in results and results["python_code"]:
            st.markdown("#### � Generated Code")
            code = results["python_code"]
            editor_key = f"code_editor_{query_id}"
            
            # The editable text area for the code
            modified_code = st.text_area(
                "You can edit the code and rerun the analysis:", 
                value=code, 
                height=300, 
                key=editor_key
            )
            
            # Rerun button
            if st.button("🚀 Rerun with Changes", key=f"rerun_{query_id}"):
                with st.spinner("Executing modified code..."):
                    df = get_current_dataframe()
                    if df is not None and modified_code:
                        # Convert dataframe to CSV string for the executor
                        df_str = df.to_csv(index=False)
                        execution_result = execute_python_code(modified_code, df_str)
                        
                        if execution_result.get("success"):
                            st.success("✅ Modified code executed successfully!")
                            # Save the successful modification to AI memory
                            if st.session_state.chat_history:
                                original_question = st.session_state.chat_history[-1]["question"]
                                save_user_logic(original_question, modified_code)
                            
                            # Display the new result
                            result_data = execution_result.get("result", {})
                            if isinstance(result_data, dict) and result_data.get("type") == "dataframe":
                                new_df = pd.DataFrame(result_data["data"])
                                st.subheader("New Result")
                                auto_visualize_dataframe(new_df)
                            else:
                                st.text(str(result_data))
                        else:
                            st.error(f"Error in modified code: {execution_result.get('error')}")
                    else:
                        st.error("No data loaded or code is empty")
        
        # Show raw results for debugging
        if st.checkbox("Show Raw Results", key=f"raw_results_{query_id}"):
            st.json(results)


def main():
    """Main application function."""
    st.title("🤖 AI Copilot for Data Teams")
    st.markdown(
        "Upload your data, then ask questions in plain English to perform complex data analysis tasks."
    )
    
    # Initialize state and AI memory
    if "initialized" not in st.session_state:
        initialize_session_state()
        initialize_memory() # Initialize the SQLite DB for learning
        # --- PERSISTENT MEMORY: Load chat history from local storage ---
        try:
            stored_history = localS.getItem("chat_history")
            if stored_history:
                st.session_state.chat_history = json.loads(stored_history)
        except:
            pass  # If localStorage fails, just continue without it

    # Update dataframe info for sidebar
    if st.session_state.get("dataframe") is not None:
        df = st.session_state.dataframe
        st.session_state.df_info = {
            "rows": df.shape[0],
            "columns": df.shape[1]
        }

    # Render sidebar
    render_sidebar()
    
    # Main layout with improved structure
    st.markdown("---")
    
    # Data loader section (always visible)
    data_container = st.container()
    with data_container:
        render_data_loader()

    # Chat interface (only show if data is loaded)
    if get_session_info()["data_loaded"]:
        st.markdown("---")
        chat_container = st.container()
        with chat_container:
            render_chat_interface()
    else:
        st.info("👆 Please load a dataset above to start the conversation.")
        
        # Show some helpful information while waiting
        with st.expander("ℹ️ What can I help you with?", expanded=True):
            st.markdown("""
            Once you upload data, I can help you:
            
            - **📊 Analyze trends** - Find patterns and insights in your data
            - **🔍 Filter & search** - Query specific data points
            - **📈 Create visualizations** - Generate charts and graphs
            - **🧮 Calculate metrics** - Compute statistics and KPIs
            - **🔄 Transform data** - Clean, reshape, and process your data
            - **💡 Get insights** - Discover hidden relationships and opportunities
            
            **Supported data sources:**
            - 📁 CSV files
            - 🗄️ PostgreSQL databases
            - 🗄️ MySQL databases  
            - 🗄️ SQLite databases
            - 🗄️ MongoDB collections
            """)

    # Save chat history to local storage periodically
    if st.session_state.get("chat_history"):
        try:
            localS.setItem("chat_history", json.dumps(st.session_state.chat_history))
        except:
            pass  # If localStorage fails, just continue

if __name__ == "__main__":
    main()