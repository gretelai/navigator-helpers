def split_statements(query):
    # Split a string of multiple statements into a list of individual statements
    statements = [s.strip() for s in query.split(";")]
    statements = [s for s in statements if s != ""]
    statements = [s + ";" for s in statements if s[-1] != ";"]
    return statements
