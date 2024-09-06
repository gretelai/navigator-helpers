import os
import re
import pandas as pd
import sqlfluff
import sqlglot

from sqlalchemy import create_engine, text
from sqlvalidator.sql_validator import SQLQuery
from typing import Tuple


# Copied from https://github.com/Gretellabs/ml-research/blob/23044df605b95d45f970c3024baf0ce2bba65429/customer_pocs/databricks_text2sql_poc/databricks_compare_quality.py#L51-L73
def is_valid_sql_with_sqlglot(sql: str) ->  Tuple[bool, str]:
    try:
        sqlglot.parse_one(sql= sql)
    except Exception as e:
        return False, str(e)
    return True, None

def is_valid_sql_with_sqlquery(sql: str) -> Tuple[bool, str]:

    query = SQLQuery(sql)
    # query._validate() does not fail with ParsingError:  https://github.com/David-Wobrock/sqlvalidator/blob/9e5bb468ba8f4364715e2da7b1804caf5eaaf83c/sqlvalidator/sql_validator.py#L28
    try:
        query._validate()
    except Exception as e:
        return False, str(e)
    if len(query.errors) == 0:
        return True, None
    else:
        return all([error=="" for error in query.errors]), "\n".join(query.errors)

def is_valid_sql_with_sqlfluff(
        content: str, dialect: str = "ansi"
    ) -> Tuple[bool, str]:
        try:
            result = sqlfluff.lint(content, dialect=dialect)
            prs_errors = [res for res in result if res["code"].startswith("PRS")]
            error_messages = "\n".join(
                [f"{error['code']}: {error['description']}" for error in prs_errors]
            )
            decimal_pattern = re.compile(r"DECIMAL\(\d+\)")
            decimal_issues = decimal_pattern.findall(content)
            if decimal_issues:
                error_messages += "\nCustom Check: Found DECIMAL definitions without a scale, which may be incorrect."
            if error_messages:
                return False, error_messages
            return True, None
        except Exception as e:
            return False, f"Exception during SQL parsing: {str(e)[:50]}..."

def split_statements(query):
    statements = [s.strip() for s in query.split(';')]
    statements = [s for s in statements if s != '']
    statements = [s + ';' for s in statements if s[-1] != ';']
    return statements

def is_valid_sql_with_sqlite(query: str, schema: str) -> Tuple[bool, str]:

    # Create the engine for an in-memory SQLite database
    engine = create_engine('sqlite:///:memory:', echo=False)

    # Split the context into individual statements
    context_statements = split_statements(schema)

    with engine.connect() as connection:
        # Execute the CREATE TABLE statements
        try:
            for statement in context_statements:
                connection.execute(text(statement))
            # print("Tables created successfully.")
        except Exception as e:
            # print(f"Error creating tables: {e}")
            return False, f"Error creating tables: {e}"

        # Query the database
        try:
            result = connection.execute(text(query))
            # print("\nQueried Transactions:")
            # print(result)
            return True, None
        except Exception as e:
            return False, f"Error querying Transactions: {e}"


def remove_db(db_name: str, db_creds: dict):

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
    context: str,
    db_creds: dict,
) -> pd.DataFrame:
    """
    Creates a temporary db from the table metadata string, runs query on the temporary db.
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
            conn.execute(text(context))
            result = conn.execute(text(sql_query))

            # No need to return results as a dataframe for now, but may be useful later
            # results_sql = func_timeout(
            #     timeout, pd.read_sql_query, args=(sql_query, engine)
            # )

        engine.dispose()  # close connection

        return result

    except Exception as e:
        if engine:
            engine.dispose()
        if admin_engine:
            admin_engine.dispose()
        if conn:
            conn.close()
        remove_db(db_name, db_creds)
        raise e

def is_valid_sql_with_postgres(query: str, context: str, domain: str, db_creds: dict) -> Tuple[bool, str]:
    db_name = domain.replace(' ','_').lower()
    try:
        query_postgres(sql_query=query,
                       db_name=db_name,
                       context=context,
                       db_creds=db_creds)
        return True, None
    except Exception as e:
        print(f"PostgreSQL Error: {e}")
        return False, str(e)
    finally:
        try:
            remove_db(db_name, db_creds)
        except:
            print('Unable to remove db')
            pass
