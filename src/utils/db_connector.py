"""
Utilities for connecting to various databases and fetching data into a pandas DataFrame.
Enhanced to automatically discover and analyze all tables in databases.
"""
import streamlit as st
import pandas as pd
import logging
from sqlalchemy import create_engine, text, inspect
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("gabi.utils.db_connector")

def get_database_schema(db_type: str, conn_details: dict) -> Dict[str, List[str]]:
    """
    Get the schema information for all tables in the database.
    
    Args:
        db_type: The type of database ('postgresql', 'mysql', 'sqlite').
        conn_details: A dictionary with connection details.
    
    Returns:
        Dictionary mapping table names to their column lists.
    """
    try:
        if db_type == 'sqlite':
            uri = f"sqlite:///{conn_details['database']}"
        else:
            dialect = 'postgresql+psycopg2' if db_type == 'postgresql' else 'mysql+mysqlconnector'
            user = conn_details['user']
            password = conn_details['password']
            host = conn_details['host']
            port = conn_details['port']
            database = conn_details['database']
            uri = f"{dialect}://{user}:{password}@{host}:{port}/{database}"

        engine = create_engine(uri)
        inspector = inspect(engine)
        
        schema_info = {}
        for table_name in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            schema_info[table_name] = columns
            
        return schema_info
    except Exception as e:
        logger.error(f"Failed to get schema for {db_type}: {e}")
        return {}

def connect_and_analyze_database(db_type: str, conn_details: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Connect to database, analyze all tables, and return a comprehensive overview.
    
    Args:
        db_type: The type of database ('postgresql', 'mysql', 'sqlite').
        conn_details: A dictionary with connection details.
    
    Returns:
        Tuple of (overview_dataframe, metadata_dict)
    """
    try:
        if db_type == 'sqlite':
            uri = f"sqlite:///{conn_details['database']}"
        else:
            dialect = 'postgresql+psycopg2' if db_type == 'postgresql' else 'mysql+mysqlconnector'
            user = conn_details['user']
            password = conn_details['password']
            host = conn_details['host']
            port = conn_details['port']
            database = conn_details['database']
            uri = f"{dialect}://{user}:{password}@{host}:{port}/{database}"

        with st.spinner(f"Connecting to {db_type} and analyzing database structure..."):
            engine = create_engine(uri)
            inspector = inspect(engine)
            
            # Get all table information
            tables_info = []
            metadata = {
                "database_type": db_type,
                "connection_details": {k: v for k, v in conn_details.items() if k != 'password'},
                "tables": {},
                "total_tables": 0,
                "total_columns": 0
            }
            
            with engine.connect() as connection:
                for table_name in inspector.get_table_names():
                    try:
                        # Get basic table info
                        columns_info = inspector.get_columns(table_name)
                        column_names = [col['name'] for col in columns_info]
                        column_types = [str(col['type']) for col in columns_info]
                        
                        # Get row count
                        try:
                            row_count_result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                            row_count = row_count_result.scalar()
                        except:
                            row_count = "Unknown"
                        
                        # Get sample data (first few rows)
                        try:
                            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                            sample_data = pd.read_sql(text(sample_query), connection)
                            sample_preview = sample_data.to_dict('records')
                        except:
                            sample_preview = []
                        
                        table_info = {
                            "table_name": table_name,
                            "row_count": row_count,
                            "column_count": len(column_names),
                            "columns": column_names,
                            "column_types": column_types,
                            "sample_data": sample_preview
                        }
                        
                        tables_info.append({
                            "Table": table_name,
                            "Rows": row_count,
                            "Columns": len(column_names),
                            "Column_Names": ", ".join(column_names[:5]) + ("..." if len(column_names) > 5 else ""),
                            "Primary_Types": ", ".join(set(column_types))
                        })
                        
                        metadata["tables"][table_name] = table_info
                        metadata["total_columns"] += len(column_names)
                        
                    except Exception as e:
                        logger.warning(f"Could not analyze table {table_name}: {e}")
                        tables_info.append({
                            "Table": table_name,
                            "Rows": "Error",
                            "Columns": "Error",
                            "Column_Names": "Could not access",
                            "Primary_Types": "Unknown"
                        })
            
            metadata["total_tables"] = len(tables_info)
            overview_df = pd.DataFrame(tables_info)
            
        logger.info(f"Successfully analyzed {len(tables_info)} tables from {db_type}.")
        return overview_df, metadata
        
    except Exception as e:
        logger.error(f"Failed to analyze {db_type} database: {e}")
        st.error(f"Database analysis failed: {e}")
        return pd.DataFrame(), {}

def connect_and_query_sql(db_type: str, conn_details: dict, query: str) -> pd.DataFrame | None:
    """
    Connects to a SQL database (PostgreSQL, MySQL, SQLite) and executes a query.

    Args:
        db_type: The type of database ('postgresql', 'mysql', 'sqlite').
        conn_details: A dictionary with connection details.
        query: The SQL query to execute.

    Returns:
        A pandas DataFrame with the query results, or None if connection fails.
    """
    try:
        if db_type == 'sqlite':
            # For SQLite, the 'database' is the file path
            uri = f"sqlite:///{conn_details['database']}"
        else:
            dialect = 'postgresql+psycopg2' if db_type == 'postgresql' else 'mysql+mysqlconnector'
            user = conn_details['user']
            password = conn_details['password']
            host = conn_details['host']
            port = conn_details['port']
            database = conn_details['database']
            uri = f"{dialect}://{user}:{password}@{host}:{port}/{database}"

        with st.spinner(f"Connecting to {db_type} and executing query..."):
            engine = create_engine(uri)
            with engine.connect() as connection:
                df = pd.read_sql(text(query), connection)
        logger.info(f"Successfully fetched {df.shape[0]} rows from {db_type}.")
        return df
    except Exception as e:
        logger.error(f"Failed to connect or query {db_type}: {e}")
        st.error(f"Connection failed: {e}")
        return None

def connect_and_query_mongo(conn_details: dict, collection_name: str, query: str) -> pd.DataFrame | None:
    """
    Connects to a MongoDB database and executes a query.

    Args:
        conn_details: A dictionary with connection details.
        collection_name: The name of the collection to query.
        query: The MongoDB query document.

    Returns:
        A pandas DataFrame with the query results, or None if connection fails.
    """
    try:
        from pymongo import MongoClient
        import json

        with st.spinner("Connecting to MongoDB and fetching data..."):
            uri = conn_details['uri']
            db_name = conn_details['database']

            client = MongoClient(uri)
            db = client[db_name]
            collection = db[collection_name]

            # Allow empty query to fetch all documents
            find_query = json.loads(query) if query.strip() else {}
            
            cursor = collection.find(find_query)
            df = pd.DataFrame(list(cursor))

            # MongoDB's _id is an object, convert it or drop it
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)

        logger.info(f"Successfully fetched {df.shape[0]} rows from MongoDB collection '{collection_name}'.")
        return df
    except ImportError:
        logger.error("pymongo is not installed. Please install it to use MongoDB.")
        st.error("MongoDB support requires 'pymongo'. Please install it.")
        return None
    except Exception as e:
        logger.error(f"Failed to connect or query MongoDB: {e}")
        st.error(f"MongoDB connection failed: {e}")
        return None

def get_mongo_collections(conn_details: dict) -> List[str]:
    """
    Get list of all collections in a MongoDB database.
    
    Args:
        conn_details: A dictionary with connection details.
    
    Returns:
        List of collection names.
    """
    try:
        from pymongo import MongoClient
        
        uri = conn_details['uri']
        db_name = conn_details['database']
        
        client = MongoClient(uri)
        db = client[db_name]
        
        collections = db.list_collection_names()
        return collections
    except ImportError:
        logger.error("pymongo is not installed.")
        return []
    except Exception as e:
        logger.error(f"Failed to get MongoDB collections: {e}")
        return []

def analyze_mongo_database(conn_details: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze all collections in a MongoDB database.
    
    Args:
        conn_details: A dictionary with connection details.
    
    Returns:
        Tuple of (overview_dataframe, metadata_dict)
    """
    try:
        from pymongo import MongoClient
        
        with st.spinner("Connecting to MongoDB and analyzing collections..."):
            uri = conn_details['uri']
            db_name = conn_details['database']
            
            client = MongoClient(uri)
            db = client[db_name]
            
            collections_info = []
            metadata = {
                "database_type": "mongodb",
                "connection_details": {"uri": uri, "database": db_name},
                "collections": {},
                "total_collections": 0,
                "total_documents": 0
            }
            
            for collection_name in db.list_collection_names():
                try:
                    collection = db[collection_name]
                    
                    # Get document count
                    doc_count = collection.count_documents({})
                    
                    # Get sample documents to understand structure
                    sample_docs = list(collection.find().limit(3))
                    
                    # Analyze field structure
                    all_fields = set()
                    if sample_docs:
                        for doc in sample_docs:
                            all_fields.update(doc.keys())
                    
                    field_list = list(all_fields)
                    
                    collection_info = {
                        "collection_name": collection_name,
                        "document_count": doc_count,
                        "fields": field_list,
                        "sample_documents": sample_docs
                    }
                    
                    collections_info.append({
                        "Collection": collection_name,
                        "Documents": doc_count,
                        "Fields": len(field_list),
                        "Field_Names": ", ".join(field_list[:5]) + ("..." if len(field_list) > 5 else ""),
                        "Sample_Available": len(sample_docs) > 0
                    })
                    
                    metadata["collections"][collection_name] = collection_info
                    metadata["total_documents"] += doc_count
                    
                except Exception as e:
                    logger.warning(f"Could not analyze collection {collection_name}: {e}")
                    collections_info.append({
                        "Collection": collection_name,
                        "Documents": "Error",
                        "Fields": "Error",
                        "Field_Names": "Could not access",
                        "Sample_Available": False
                    })
            
            metadata["total_collections"] = len(collections_info)
            overview_df = pd.DataFrame(collections_info)
            
        logger.info(f"Successfully analyzed {len(collections_info)} collections from MongoDB.")
        return overview_df, metadata
        
    except ImportError:
        logger.error("pymongo is not installed.")
        st.error("MongoDB support requires 'pymongo'. Please install it.")
        return pd.DataFrame(), {}
    except Exception as e:
        logger.error(f"Failed to analyze MongoDB: {e}")
        st.error(f"MongoDB analysis failed: {e}")
        return pd.DataFrame(), {}
