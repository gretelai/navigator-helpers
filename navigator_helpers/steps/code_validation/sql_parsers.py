import os
import pandas as pd
import sqlglot
import sqlite3

from func_timeout import func_timeout
from sqlalchemy import create_engine, text
from sqlvalidator.sql_validator import SQLQuery
from typing import Tuple


# Copied from https://github.com/Gretellabs/ml-research/blob/23044df605b95d45f970c3024baf0ce2bba65429/customer_pocs/databricks_text2sql_poc/databricks_compare_quality.py#L51-L73
def is_parsable_sql(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql= sql)
    except:
        return False
    return True

def is_valid_sql(sql: str) -> bool:

    query = SQLQuery(sql)
    # query._validate() does not fail with ParsingError:  https://github.com/David-Wobrock/sqlvalidator/blob/9e5bb468ba8f4364715e2da7b1804caf5eaaf83c/sqlvalidator/sql_validator.py#L28
    try:
        query._validate()
    except:
        return False
    if len(query.errors) == 0:
        return True
    else:
        return all([error=="" for error in query.errors])


def is_valid_sql_against_schema_sqlite(query: str, schema: str) -> Tuple[bool, str]:
    try:
        # Connect to an in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Apply the provided schema
        cursor.executescript(schema)
        
        # Try to execute the query
        cursor.execute(query)
        
        # If no exceptions were raised, the query is valid
        return True, None
    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
        return False, str(e)
    finally:
        conn.close()

# Copied from https://github.com/Gretellabs/ml-research/blob/master/customer_pocs/databricks_text2sql_poc/main_analyse_failed_sql.ipynb
db_creds = {
        "host": os.environ.get("DBHOST", "localhost"),
        "port": os.environ.get("DBPORT", 5432),
        "user": os.environ.get("DBUSER", "postgres"),
        "password": os.environ.get("DBPASSWORD", "postgres"),
    }

def remove_db(db_name: str):

    """
    Removes the temporary database created for the query.
    """
    admin_engine = None
    conn = None
    try:
        admin_db_url = f"postgresql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/postgres"
        admin_engine = create_engine(admin_db_url)
        with admin_engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
            conn.close()
        admin_engine.dispose()  # close connection
    except Exception as e:
        if admin_engine:
            admin_engine.dispose()
        if conn:
            conn.close()
        raise e
    
def query_postgres(
    sql_query: str,
    db_name: str,
    ddl: str = "",
    timeout: float = 10.0,
) -> pd.DataFrame:
    """
    Creates a temporary db from the table metadata string, runs query on the temporary db, and returns results as a dataframe.
    After the query is run, the temporary db is dropped.
    timeout: time in seconds to wait for query to finish before timing out
    """
    engine = None
    admin_engine = None
    conn = None

    try:
        # create a temporary database on postgres if it doesn't exist
        admin_db_url = f"postgresql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/postgres"
        admin_engine = create_engine(admin_db_url)
        with admin_engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            db_exists = (
                conn.execute(
                    text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
                ).first()
                is not None
            )
            if not db_exists:
                conn.execute(text(f"CREATE DATABASE {db_name}"))
            conn.close()
        admin_engine.dispose()  # close connection

        # create tables in the temporary database and execute sql query
        db_url = f"postgresql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text(ddl))

            results_sql = func_timeout(
                timeout, pd.read_sql_query, args=(sql_query, engine)
            )

        engine.dispose()  # close connection

        return results_sql #,results_pd

    except Exception as e:
        if engine:
            engine.dispose()
        if admin_engine:
            admin_engine.dispose()
        if conn:
            conn.close()
        remove_db(db_name)
        raise e

def is_valid_sql_against_schema_postgres(query: str, schema: str, domain: str) -> Tuple[bool, str]:
    db_name = domain.replace(' ','_').lower()
    try:
        query_postgres(sql_query=query,
                       db_name=db_name,
                       ddl=schema,)
        return True, None
    except Exception as e:
        print(f"PostgreSQL Error: {e}")
        return False, str(e)
    finally:
        try:
            remove_db(db_name)
        except:
            print('Unable to remove db')
            pass
