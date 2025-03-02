from datetime import datetime
import sqlite3

def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    with conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                age INTEGER,
                experience INTEGER,
                income REAL,
                family INTEGER,
                cc_avg REAL,
                education INTEGER,
                mortgage REAL,
                securities_account INTEGER,
                cd_account INTEGER,
                online INTEGER,
                credit_card INTEGER,
                prediction INTEGER,
                prediction_time TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        ''')