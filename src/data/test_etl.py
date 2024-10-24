from data.etl import run_etl

if __name__ == "__main__":
    try:
        run_etl()
        print("ETL pipeline executed successfully.")
    except Exception as e:
        print(f"ETL pipeline failed with error: {e}")