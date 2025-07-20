
import sqlite3
import pandas as pd
import os

def explore_database(db_path):
    """
    Connects to the SQLite database, prints all table names,
    and the schema of each table.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        return

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
            return

        print("Tables in the database:")
        for table in tables:
            table_name = table[0]
            print(f"- {table_name}")

            # Get the schema of each table using pandas
            print(f"\nSchema for table '{table_name}':")
            # Use a parameterized query to prevent SQL injection, although less critical here
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
            print(f"Columns: {df.columns.tolist()}")
            print("Data summary:")
            print(df.info())
            print("First 5 rows:")
            print(df.head())


    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Relative path to the database file from Data directory
    db_path = 'joined_data/merged_data.db'
    explore_database(db_path) 