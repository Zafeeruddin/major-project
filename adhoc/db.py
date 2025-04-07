import psycopg2
from psycopg2 import sql

# Database connection parameters
db_params = {
    'dbname': 'major_project',
    'user': 'postgres',
    'password': 'mysecretpassword',
    'host': 'localhost',
    'port': '5432'
}

# Connect to the default database to create the new database if it doesn't exist
default_conn = psycopg2.connect(dbname='postgres', user=db_params['user'], password=db_params['password'], host=db_params['host'], port=db_params['port'])
default_conn.autocommit = True
default_cur = default_conn.cursor()

# SQL command to create the database
create_db_query = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_params['dbname']))

# Try to create the database
try:
    default_cur.execute(create_db_query)
    print(f"Database '{db_params['dbname']}' created successfully.")
except psycopg2.errors.DuplicateDatabase:
    print(f"Database '{db_params['dbname']}' already exists.")

# Close the default connection
default_cur.close()
default_conn.close()

# Connect to the newly created or existing database
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# SQL command to create the table
create_table_query = '''
CREATE TABLE IF NOT EXISTS footfall_data (
    datetime TIMESTAMP PRIMARY KEY,
    footfall INTEGER,
    is_holiday BOOLEAN,
    day_of_week INTEGER,
    is_weekend BOOLEAN
);
'''

# Execute the SQL command
cur.execute(create_table_query)

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()

print("Table created successfully.")