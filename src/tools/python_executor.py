"""
Tool for safely executing Python (Pandas) code on a DataFrame.
"""

from typing import Dict, Any
import pandas as pd
import io
import logging

logger = logging.getLogger("gabi.tools")

def execute_python_code(code: str, dataframe_str: str) -> Dict[str, Any]:
    """
    Execute Python code with a pre-loaded DataFrame.

    Args:
        code: The Python code to execute.
        dataframe_str: The CSV string representation of the DataFrame.

    Returns:
        A dictionary with the execution result or an error.
    """
    try:
        # Parse the CSV string with more robust error handling
        logger.debug(f"Attempting to parse CSV data. First 200 characters: {dataframe_str[:200]}")
        
        try:
            # Try with standard CSV parsing first
            df = pd.read_csv(io.StringIO(dataframe_str))
            logger.debug(f"Successfully parsed CSV. DataFrame shape: {df.shape}")
        except pd.errors.ParserError as e:
            logger.warning(f"Standard CSV parsing failed: {e}. Trying with different parameters.")
            # Try with different parsing options
            try:
                df = pd.read_csv(io.StringIO(dataframe_str), sep=',', quotechar='"', escapechar='\\')
                logger.debug(f"Successfully parsed CSV with custom parameters. DataFrame shape: {df.shape}")
            except pd.errors.ParserError:
                # Try with error handling for bad lines
                try:
                    df = pd.read_csv(io.StringIO(dataframe_str), on_bad_lines='skip')
                    logger.debug(f"Successfully parsed CSV with bad lines skipped. DataFrame shape: {df.shape}")
                except (TypeError, ValueError):
                    # Final fallback: try to parse line by line or use a more permissive approach
                    logger.error("All CSV parsing attempts failed. Using manual parsing.")
                    raise ValueError("Unable to parse CSV data")

        # Prepare the execution environment
        local_vars = {"df": df}
        global_vars = {
            "pd": pd,
            # Add commonly used functions to avoid dependency issues
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
        }

        # Log the code being executed for debugging
        logger.debug(f"Executing Python code:\n{code}")

        # First, try to compile the code to catch syntax errors early
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as se:
            logger.error(f"Syntax error in Python code: {se}")
            logger.error(f"Problematic code:\n{code}")
            return {"success": False, "error": f"Syntax error: {se}"}

        # Capture stdout to get print() output
        old_stdout = io.StringIO()
        import sys
        sys.stdout = old_stdout

        try:
            # Execute the code
            exec(code, global_vars, local_vars)
        finally:
            # Restore stdout
            captured_output = old_stdout.getvalue()
            sys.stdout = sys.__stdout__

        # Retrieve the result
        result = local_vars.get("result")

        # Handle different result types for serialization
        if isinstance(result, pd.DataFrame):
            # Convert DataFrame to a serializable format
            output = {
                "type": "dataframe",
                "data": result.to_dict('records'),
                "columns": result.columns.tolist(),
                "shape": result.shape,
                "dtypes": result.dtypes.astype(str).to_dict()
            }
        elif isinstance(result, pd.Series):
            # Convert Series to a serializable format
            series_df = result.to_frame().reset_index()
            output = {
                "type": "dataframe", 
                "data": series_df.to_dict('records'),
                "columns": series_df.columns.tolist(),
                "shape": series_df.shape,
                "dtypes": series_df.dtypes.astype(str).to_dict()
            }
        elif result is None and captured_output.strip():
            # If no result variable but we have print output
            output = captured_output.strip()
        elif result is None:
            output = "Code executed successfully, but no result was returned."
        elif isinstance(result, dict):
            # Return dictionary as-is for structured data (like visualization configs)
            output = result
        else:
            output = str(result)
            
        return {"success": True, "result": output, "output": captured_output.strip()}

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error executing Python code: {e}")
        
        # Provide helpful hints for common errors
        if "tabulate" in error_msg:
            error_msg += "\nHint: Try using df.head() instead of methods that require tabulate"
        elif "missing" in error_msg.lower() and "dependency" in error_msg.lower():
            error_msg += "\nHint: Some pandas display features may not be available. Try simpler operations."
        
        return {"success": False, "error": error_msg}
