"""
Configuration settings for the AI Copilot for Data Teams.
"""

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Result limits
MAX_RESULTS_DISPLAY = 50

# Memory configuration
MEMORY_CHECKPOINT_NAME = "data-copilot-memory"

EXAMPLE_QUERIES = {
    "Code Generation": [
        "Show me the first 10 rows of the dataset",
        "Plot a histogram of the 'age' column",
        "Group by 'department' and calculate the average 'salary'",
        "Filter the data to only include rows where 'status' is 'active'",
    ],
    "Data Quality": [
        "Generate a data quality report",
        "Are there any missing values in this dataset?",
        "Check for duplicate rows",
        "Identify potential outliers in the 'sales' column",
    ],
    "Summarization & KPIs": [
        "Summarize this dataset for me",
        "What are the main characteristics of this data?",
        "Suggest some relevant KPIs for this dataset",
        "What key metrics should I track based on these columns?",
    ],
    "General Help": [
        "Hello, what can you help me with?",
        "How does this application work?",
    ],
}
