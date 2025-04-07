import psycopg2
import csv

def connect_db():
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(
        dbname='major_project',
        user='postgres',
        password='mysecretpassword',
        host='localhost',  # Use the container's IP if localhost doesn't work
        port='5432'
    )

def export_to_csv():
    conn = connect_db()
    cursor = conn.cursor()
    query = "SELECT * FROM footfall_data;"
    cursor.execute(query)

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Get column names
    colnames = [desc[0] for desc in cursor.description]

    # Write to CSV
    with open('footfall_data.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(colnames)  # Write column headers
        csvwriter.writerows(rows)     # Write data rows

    # Close the cursor and connection
    cursor.close()
    conn.close()

if __name__ == "__main__":
    export_to_csv()
    print("Data exported to footfall_data.csv")