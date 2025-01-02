import os
import sqlite3
from typing import Optional
import numpy as np

class DataManager:
    def __init__(self, db_path: str = "data/database.sqlite"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                image_path TEXT,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    def save_model(self, prompt: str, image_path: Optional[str], model_path: str) -> int:
        """Save generated model information to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if image_path is None:
            image_path = ""
        
        cursor.execute("""
            INSERT INTO models (prompt, image_path, model_path)
            VALUES (?, ?, ?)
        """, (prompt, image_path, model_path))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return model_id
