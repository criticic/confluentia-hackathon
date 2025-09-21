"""
Persistent memory for the AI Copilot.

This module provides a simple mechanism for the agent to "learn" from user
modifications by storing successful user-edited code snippets and retrieving
them for similar future questions.
"""
import sqlite3
import logging
from typing import Optional

logger = logging.getLogger("gabi.core.memory")

DB_PATH = "copilot_memory.db"

def initialize_memory():
    """Initializes the SQLite database for memory."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_logic (
                    id INTEGER PRIMARY KEY,
                    original_question TEXT NOT NULL UNIQUE,
                    modified_code TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info("AI memory database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing AI memory database: {e}")

def save_user_logic(question: str, code: str):
    """
    Saves or updates a user's modified code for a given question.
    If the question already exists, it updates the code and increments the usage count.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learned_logic (original_question, modified_code)
                VALUES (?, ?)
                ON CONFLICT(original_question) DO UPDATE SET
                    modified_code = excluded.modified_code,
                    usage_count = usage_count + 1,
                    last_used = CURRENT_TIMESTAMP
            """, (question.strip(), code.strip()))
            conn.commit()
        logger.info(f"Saved user logic for question: '{question[:50]}...'")
    except Exception as e:
        logger.error(f"Error saving user logic to AI memory: {e}")

def get_learned_logic(question: str) -> Optional[str]:
    """
    Retrieves learned logic for a given question.
    (This is a simple exact-match lookup for now. A more advanced version would use similarity search).
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT modified_code FROM learned_logic WHERE original_question = ?", (question.strip(),))
            result = cursor.fetchone()
            if result:
                logger.info(f"Retrieved learned logic for question: '{question[:50]}...'")
                return result[0]
        return None
    except Exception as e:
        logger.error(f"Error retrieving learned logic from AI memory: {e}")
        return None
