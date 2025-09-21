# AI Copilot for Data Teams

This project is an AI-powered assistant designed to streamline the day-to-day operations of data teams. It acts as a "copilot" by automating repetitive tasks, generating code, providing data quality insights, and creating visualizations from natural language prompts. Users can upload a CSV dataset and interact with the copilot to perform complex data analysis tasks effortlessly.

Developed for the **Confluentia Hackathon (PS 3)**.

Demo Link: [https://confluentia-hackathon-d426.streamlit.app/](https://confluentia-hackathon-d426.streamlit.app/)

## Features

- **Dynamic Data Source**: Upload your own CSV file for analysis.
- **Dynamic Example Queries**: Get personalized example queries automatically generated based on your specific dataset's schema and content.
- **Natural Language Interaction**: Ask questions and give commands in plain English.
- **Code Generation**: Automatically generate Python (Pandas) code for data manipulation and analysis.
- **Data Quality Analysis**: Detect and get reports on missing values, outliers, and duplicates.
- **Automated Visualizations**: Instantly generate relevant charts and graphs from your data.
- **Dataset Summarization**: Get a quick, plain-English summary of your dataset's contents and structure.
- **KPI Suggestion**: Receive suggestions for relevant Key Performance Indicators (KPIs) based on your data.
- **Intelligent Task Routing**: An agentic workflow routes your request to the correct specialist agent (e.g., code generator, data quality analyst).
- **Follow-up Analysis**: Smart routing system that can automatically perform additional analysis based on agent responses.
- **Interactive Web Interface**: A user-friendly Streamlit interface for uploading data and chatting with the copilot.
- **Secure Code Execution**: A sandboxed environment for executing generated Python code, ensuring that no harmful commands are run.
- **LLM Agnostic**: Built with Langchain, allowing for easy swapping between different LLM providers.

## Setup

1. First, install the required packages using [`uv`](https://docs.astral.sh/uv/):

    ```bash
    uv pip install -r requirements.txt
    ```

2. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and create a `.env` file in the root directory of the project with the following content:

    ```env
    GEMINI_API_KEY=<your_gemini_api_key>
    ```

3. Finally, run the app using:

    ```bash
    uv run streamlit run src/web/app.py
    ```

## LangGraph Architecture

The application uses a graph-based agentic workflow built with LangGraph. When a user submits a query, it's first sent to a **Task Router** agent. This router analyzes the user's intent and the available data schema to decide which specialized agent is best suited for the job.

The request is then routed to one of the following agents:

- **Python Generator**: For requests that require data manipulation, analysis, or complex calculations, this agent writes and executes Python (Pandas) code.
- **Data Quality Agent**: For requests related to data cleaning and validation, this agent generates a comprehensive report on missing values, duplicates, and outliers.
- **Summarizer Agent**: For general requests about the dataset's content, this agent provides a high-level summary.
- **KPI Suggester Agent**: If the user needs inspiration for what to analyze, this agent suggests relevant metrics and Key Performance Indicators (KPIs).
- **Follow-up Router**: Analyzes agent responses to determine if additional analysis should be performed automatically.

The workflow also includes specialized nodes for:

- **Python Executor**: Safely executes generated Python code in a sandboxed environment.
- **Visualization Generator**: Creates appropriate chart configurations based on analysis results.
- **Results Explainer**: Provides natural language explanations of analysis results.

Additionally, when users upload CSV files, the system automatically generates personalized example queries using the **Dynamic Query Generator**, which analyzes the dataset schema and sample data to create relevant, actionable queries specific to the uploaded data.

The output from the selected agent (e.g., a dataframe, a text report, or a visualization) is then returned to the user.

## How It Works

1. **Upload Your Data**: Start by uploading a CSV file through the web interface.

2. **Get Personalized Examples**: The system automatically analyzes your data and generates relevant example queries tailored to your specific dataset's columns and content.

3. **Ask Questions**: Use the generated examples or ask your own questions in natural language about your data.

4. **Intelligent Routing**: The Task Router analyzes your question and routes it to the most appropriate specialist agent.

5. **Automated Analysis**: The selected agent performs the analysis, generates code, creates visualizations, or provides insights as needed.

6. **Interactive Results**: View results with explanations, code snippets, visualizations, and the ability to download or further analyze the data.

7. **Follow-up Analysis**: The system can automatically suggest and perform additional analysis based on initial results.

## Project Structure

```plaintext
.
├── LICENSE
├── PROBLEM-STATEMENT.md
├── README.md
├── __init__.py
├── main.py
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── agents.py
│   │   └── graph.py
│   ├── models
│   │   ├── __init__.py
│   │   └── gemini.py
│   ├── prompts
│   │   └── __init__.py
│   ├── tools
│   │   ├── __init__.py
│   │   └── python_executor.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── logging.py
│   └── web
│       ├── __init__.py
│       ├── app.py
│       ├── components
│       │   ├── __init__.py
│       │   ├── chat.py
│       │   ├── sidebar.py
│       │   └── visualization.py
│       ├── handlers.py
│       ├── run.py
│       └── state.py
└── uv.lock
```
