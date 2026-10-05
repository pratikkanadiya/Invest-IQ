import os
from psycopg import connect
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    return connect(DATABASE_URL)