from sqlalchemy import create_engine, text
import pandas as pd


DB_URI = "sqlite:///instance//database.db"
# SQL ALCHEMY to PANDAS data class


class SQL:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri, echo=True)
        self.conn = self.engine.connect()

    def get_table_data(self, table_name):
        df = pd.read_sql_table(table_name, self.conn)
        return df

    def get_sql_query(self, query):
        df = pd.read_sql_query(query, self.conn)
        return df

    def get_all_table_names(self):
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        for table_name in inspector.get_table_names():
            for column in inspector.get_columns(table_name):
                print("Column: %s" % column["name"])

    def __repr__(self):
        print(f"{self.db_uri}")


if __name__ == "__main__":
    sql = SQL(DB_URI)
    sql.__repr__()
