import os
import time
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# 1. Load variables from .env file
load_dotenv()

# 2. Get the URL
# SECURITY FIX: We removed the hardcoded fallback. 
# If DATABASE_URL is missing, this returns None.
DATABASE_URL = os.getenv("DATABASE_URL")

# 3. Validate Configuration
if not DATABASE_URL:
    raise ValueError("❌ Error: DATABASE_URL is missing! Check your .env file.")

# 4. Create the Engine
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    """Creates the tables with a retry loop."""
    retries = 5
    while retries > 0:
        try:
            SQLModel.metadata.create_all(engine)
            print("✅ Database: Connected & Tables Ready.")
            return
        except OperationalError as e:
            print(f"⏳ Database: Connection failed. Retrying in 2s... ({retries} left)")
            # Helpful debugging (prints the error without leaking the password)
            print(f"   -> Details: {str(e).split('@')[-1]}") 
            time.sleep(2)
            retries -= 1
    
    print("❌ Database: Could not connect. Is Postgres running?")

def get_session():
    """Returns a database session."""
    return Session(engine)