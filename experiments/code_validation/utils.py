import re


def split_statements(query):
    # Split a string of multiple statements into a list of individual statements
    statements = [s.strip() for s in query.split(";")]
    statements = [s for s in statements if s != ""]
    statements = [s + ";" for s in statements if s[-1] != ";"]
    return statements


def create_db_name(domain):
    # Given a domain name, create a valid database name
    db_name = domain.replace(" ", "_").replace("-", "_").lower()
    # If the db_name starts with a digit, prepend "db_"
    if bool(re.match(r"^\d", db_name)):
        db_name = "db_" + db_name
    return db_name
