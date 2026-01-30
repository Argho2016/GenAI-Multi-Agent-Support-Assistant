import sqlite3
import os
import pandas as pd

DB_PATH = "data/customers.db"

CUSTOMERS = [
    {"customer_id": 1, "name": "Ema Stone", "email": "ema.stone@example.com", "plan": "Premium", "region": "Canada"},
    {"customer_id": 2, "name": "Liam Park", "email": "liam.park@example.com", "plan": "Standard", "region": "USA"},
    {"customer_id": 3, "name": "Noah Khan", "email": "noah.khan@example.com", "plan": "Basic", "region": "UK"},
]

TICKETS = [
    {"ticket_id": 101, "customer_id": 1, "created_at": "2025-11-02", "topic": "Refund request", "status": "Closed", "resolution": "Refund issued"},
    {"ticket_id": 102, "customer_id": 1, "created_at": "2025-12-14", "topic": "Login issue", "status": "Closed", "resolution": "Password reset"},
    {"ticket_id": 201, "customer_id": 2, "created_at": "2025-10-20", "topic": "Billing question", "status": "Open", "resolution": ""},
    {"ticket_id": 301, "customer_id": 3, "created_at": "2025-09-07", "topic": "Plan downgrade", "status": "Closed", "resolution": "Downgraded"},
]

def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        plan TEXT,
        region TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        created_at TEXT,
        topic TEXT,
        status TEXT,
        resolution TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    """)

    cur.execute("DELETE FROM tickets;")
    cur.execute("DELETE FROM customers;")

    pd.DataFrame(CUSTOMERS).to_sql("customers", conn, if_exists="append", index=False)
    pd.DataFrame(TICKETS).to_sql("tickets", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"Seeded DB at {DB_PATH}")

if __name__ == "__main__":
    main()
