import re
import time

from typing import Tuple

import pandas as pd
import sqlfluff
import sqlglot

from docker.models.containers import Container
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.auth.credentials import AnonymousCredentials
from google.cloud import bigquery
from sqlalchemy import create_engine, text
from sqlvalidator.sql_validator import SQLQuery
from utils import split_statements


class SimpleSqlValidator:
    # Copied from https://github.com/Gretellabs/ml-research/blob/23044df605b95d45f970c3024baf0ce2bba65429/customer_pocs/databricks_text2sql_poc/databricks_compare_quality.py#L51-L73
    def is_valid_sql_with_sqlglot(sql: str) -> Tuple[bool, str]:
        try:
            sqlglot.parse_one(sql=sql)
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
            return all([error == "" for error in query.errors]), "\n".join(query.errors)

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
            return False, f"Exception during SQL parsing: {str(e)}"


class SqliteValidator:

    def is_valid_sql(query: str, schema: str) -> Tuple[bool, str]:

        # Create the engine for an in-memory SQLite database
        engine = create_engine("sqlite:///:memory:", echo=False)

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


class PostgresqlValidator:

    def _remove_db(db_name: str, db_creds: dict):
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

    def _query_postgres(
        sql_query: str,
        schema: str,
        db_name: str,
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
                conn.execute(text(schema))
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
            PostgresqlValidator._remove_db(db_name, db_creds)
            raise e

    def is_valid_sql(
        query: str, schema: str, domain: str, db_creds: dict
    ) -> Tuple[bool, str]:
        db_name = domain.replace(" ", "_").replace("-", "_").lower()
        try:
            PostgresqlValidator._query_postgres(
                sql_query=query, schema=schema, db_name=db_name, db_creds=db_creds
            )
            return True, None
        except Exception as e:
            # print(f"PostgreSQL Error: {e}")
            return False, str(e)
        finally:
            try:
                PostgresqlValidator._remove_db(db_name, db_creds)
            except:
                # print('Unable to remove db')
                pass


class MysqlValidator:
    def _create_db(db_name: str, db_creds: dict, mysql_container: Container):
        mysql_command = f"mysql -u {db_creds['user']} -p'{db_creds['password']}' -e \"CREATE DATABASE IF NOT EXISTS {db_name};\""
        exit_code, output = mysql_container.exec_run(mysql_command)
        # print(f"exit_code: {exit_code}, output: {output}")
        assert exit_code == 0, "Failed to create database"

    def _remove_db(db_name: str, db_creds: dict, mysql_container: Container):
        mysql_command = f"mysql -u {db_creds['user']} -p'{db_creds['password']}' -e \"DROP DATABASE IF EXISTS {db_name};\""
        exit_code, output = mysql_container.exec_run(mysql_command)
        # print(f"exit_code: {exit_code}, output: {output}")
        assert exit_code == 0, "Failed to drop database"

    def _query_mysql(
        sql_query: str,
        schema: str,
        db_name: str,
        db_creds: dict,
    ) -> pd.DataFrame:
        """
        Creates a temporary db from the table metadata string, runs query on the temporary db.
        After the query is run, the temporary db is dropped.
        """
        engine = None
        conn = None

        try:
            # create tables in the temporary database and execute sql query
            db_url = f"mysql+mysqlconnector://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
            engine = create_engine(db_url)

            # Split the context into individual statements
            context_statements = split_statements(schema)

            with engine.connect() as conn:
                for statement in context_statements:
                    conn.execute(text(statement))
                result = conn.execute(text(sql_query))

            engine.dispose()  # close connection

            return result

        except Exception as e:
            if engine:
                engine.dispose()
            if conn:
                conn.close()
            raise e

    def is_valid_sql(
        query: str, schema: str, domain: str, db_creds: dict, mysql_container: Container
    ) -> Tuple[bool, str]:
        db_name = domain.replace(" ", "_").replace("-", "_").lower()
        try:
            # A MySQL database need to be created before creating an engine
            MysqlValidator._create_db(db_name, db_creds, mysql_container)
            MysqlValidator._query_mysql(
                sql_query=query, schema=schema, db_name=db_name, db_creds=db_creds
            )
            return True, None
        except Exception as e:
            # print(f"MySQL Error: {e}")
            return False, str(e)
        finally:
            try:
                MysqlValidator._remove_db(db_name, db_creds, mysql_container)
            except:
                # print('Unable to remove db')
                pass


class SqlserverValidator:
    def _create_db(db_name: str, db_creds: dict, sqlserver_container: Container):
        sqlserver_command = f"/opt/mssql-tools/bin/sqlcmd -S {db_creds['host']},{db_creds['port']} -U {db_creds['user']} -P '{db_creds['password']}' -Q \"CREATE DATABASE {db_name};\""
        exit_code, output = sqlserver_container.exec_run(sqlserver_command)
        # print(f"exit_code: {exit_code}, output: {output}")
        assert exit_code == 0, "Failed to create database"

    def _remove_db(db_name: str, db_creds: dict, sqlserver_container: Container):
        sqlserver_command = f"/opt/mssql-tools/bin/sqlcmd -S {db_creds['host']},{db_creds['port']} -U {db_creds['user']} -P '{db_creds['password']}' -Q \"DROP DATABASE {db_name};\""
        exit_code, output = sqlserver_container.exec_run(sqlserver_command)
        # print(f"exit_code: {exit_code}, output: {output}")
        assert exit_code == 0, "Failed to drop database"

    def _query_sqlserver(
        sql_query: str, schema: str, db_name: str, db_creds: dict
    ) -> pd.DataFrame:
        """
        Creates a temporary db from the table metadata string, runs query on the temporary db.
        After the query is run, the temporary db is dropped.
        """
        engine = None
        conn = None

        try:
            # create tables in the temporary database and execute sql query
            db_url = f"mssql+pyodbc://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
            engine = create_engine(db_url)

            with engine.connect() as conn:
                conn.execute(text(schema))
                result = conn.execute(text(sql_query))

            engine.dispose()  # close connection

            return result

        except Exception as e:
            if engine:
                engine.dispose()
            if conn:
                conn.close()
            raise e

    def is_valid_sql(
        query: str,
        schema: str,
        domain: str,
        db_creds: dict,
        sqlserver_container: Container,
    ) -> Tuple[bool, str]:
        db_name = domain.replace(" ", "_").replace("-", "_").lower()
        try:
            # A SQL Server database need to be created before creating an engine
            SqlserverValidator._create_db(db_name, db_creds, sqlserver_container)
            SqlserverValidator._query_sqlserver(
                sql_query=query, schema=schema, db_name=db_name, db_creds=db_creds
            )
            return True, None
        except Exception as e:
            # print(f"SQL Server Error: {e}")
            return False, str(e)
        finally:
            try:
                SqlserverValidator._remove_db(db_name, db_creds, sqlserver_container)
            except:
                # print('Unable to remove db')
                pass


class GooglesqlValidator:

    def _add_dataset_name_to_create_statement(create_statement, dataset_name):
        """
        This is a hack. Most BigQuery SQL statements do not include the dataset name
        in the CREATE TABLE statement. This function adds the dataset name to the table name
        that's being created so that we can query from multiple datasets within the same database instance.
        """
        # Regex to match the table name in the CREATE TABLE/CREATE VIEW/INSERT INTO statements
        pattern_1 = r"(CREATE\s+TABLE\s+)(\w+)(\s*\()"
        pattern_2 = r"(CREATE\s+VIEW\s+|INSERT\s+INTO\s+)(\w+)(\s*)"

        # Replace with the dataset and table name in the format `dataset_name.table_name`
        replacement = r"\1`" + dataset_name + r".\2`\3"

        # Apply the regex substitution
        updated_statement = re.sub(pattern_1, replacement, create_statement)
        updated_statement = re.sub(pattern_2, replacement, updated_statement)

        return updated_statement

    def _add_dataset_name_to_select_statement(query, dataset_name):
        """Similar hack as above, adds dataset name to table names in FROM and JOIN clauses"""

        # Step 1: Edge case handling
        # Find all EXTRACT(...FROM...) patterns and temporarily replace them to avoid modification
        extract_pattern = r"EXTRACT\s*\(\s*\w+\s+FROM\s+[\w.\(]+\s*\)"
        extracts = re.findall(extract_pattern, query, flags=re.IGNORECASE)

        # Temporarily replace EXTRACT(...FROM...) with placeholders
        placeholder_query = query
        for i, extract in enumerate(extracts):
            placeholder_query = placeholder_query.replace(
                extract, f"__EXTRACT_PLACEHOLDER_{i}__"
            )

        # Step 2: Add dataset name to table names in FROM and JOIN clauses
        pattern = r"(?<=\bFROM\s|\bJOIN\s)(\w+)\b"
        updated_query = re.sub(
            pattern, rf"{dataset_name}.\1", placeholder_query, flags=re.IGNORECASE
        )

        # Step 3: Restore the original EXTRACT(...) functions
        for i, extract in enumerate(extracts):
            updated_query = updated_query.replace(
                f"__EXTRACT_PLACEHOLDER_{i}__", extract
            )

        return updated_query

    def _delete_views(dataset_id, client):
        dataset_ref = client.dataset(dataset_id)

        # List all tables in the dataset
        tables = client.list_tables(dataset_ref)

        # Loop through each table to check if it's a view
        for table in tables:
            table_ref = client.get_table(table)
            if table_ref.table_type == "VIEW":
                # Try to delete the view
                try:
                    client.delete_table(table_ref)
                except NotFound:
                    pass
                except Exception as e:
                    print(f"Failed to delete view: {table.table_id}, error: {e}")

    def _query_bigquery(
        sql_query: str, schema: str, db_name: str, db_creds: dict
    ) -> pd.DataFrame:
        """
        Creates a temporary db from the table metadata string, runs query on the temporary db.
        After the query is run, the temporary db is dropped.
        """
        client = None

        try:
            # Create a client
            client_options = ClientOptions(
                api_endpoint=f"http://0.0.0.0:{db_creds['port']}"
            )
            client = bigquery.Client(
                project=db_creds["project"],
                client_options=client_options,
                credentials=AnonymousCredentials(),  # Auth is disabled for our purposes
            )

            # Create a dataset
            client.create_dataset(db_name, exists_ok=True, retry=None)
            # Wait for the dataset to be created
            time.sleep(1)
        except Exception as e:
            if client:
                client.close()
            raise RuntimeError(f"Error creating dataset: {e}")

        try:
            # Create a table
            disambiguated_schema = (
                GooglesqlValidator._add_dataset_name_to_create_statement(
                    schema, db_name
                )
            )
            disambiguated_schema = (
                GooglesqlValidator._add_dataset_name_to_select_statement(
                    disambiguated_schema, db_name
                )
            )
            schema_creation = client.query(disambiguated_schema)
            schema_creation.result()

        except Exception as e:
            if client:
                try:
                    GooglesqlValidator._delete_views(db_name, client)
                    client.delete_dataset(
                        db_name, delete_contents=True, not_found_ok=True, retry=None
                    )
                except:
                    pass
                client.close()
            raise RuntimeError(f"Error creating tables: {e}")

        try:
            # Execute the query
            disambiguated_query = (
                GooglesqlValidator._add_dataset_name_to_select_statement(
                    sql_query, db_name
                )
            )
            query_job = client.query(disambiguated_query, retry=None)
            result = query_job.result()
        except Exception as e:
            if client:
                try:
                    GooglesqlValidator._delete_views(db_name, client)
                    client.delete_dataset(
                        db_name, delete_contents=True, not_found_ok=True, retry=None
                    )
                except:
                    pass
                client.close()
            raise RuntimeError(f"Error querying transactions: {e}")

        try:
            # Delete the dataset
            GooglesqlValidator._delete_views(db_name, client)
            client.delete_dataset(
                db_name, delete_contents=True, not_found_ok=True, retry=None
            )
            client.close()
        except:
            pass

        return result

    def is_valid_sql(
        query: str, schema: str, domain: str, db_creds: dict
    ) -> Tuple[bool, str]:
        db_name = domain.replace(" ", "-").replace("_", "-").lower()
        try:
            GooglesqlValidator._query_bigquery(
                sql_query=query, schema=schema, db_name=db_name, db_creds=db_creds
            )
            return True, None
        except Exception as e:
            # print(f"GoogleSQL Error: {e}")
            return False, str(e)
