from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import dotenv

dotenv.load_dotenv()

user = os.environ.get("MYSQL_USER")
password = os.environ.get("MYSQL_PASSWORD")
server = os.environ.get("MYSQL_SERVER")
db = os.environ.get("MYSQL_DB")

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{user}:{password}@{server}/{db}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()