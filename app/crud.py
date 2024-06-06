from sqlalchemy.orm import Session
from .security import hash_password
from . import models, schemas
# import models
# import schemas
import time


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = hash_password(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_conversations(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Conversation).offset(skip).limit(limit).all()

def get_user_conversations(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Conversation).filter(models.Conversation.user_id == user_id) \
        .order_by(models.Conversation.created_at.desc()) \
        .offset(skip).limit(limit).all()

def create_conversation(db: Session, conversation: schemas.ConversationCreate, user_id: int):
    db_conversation = models.Conversation(**conversation, user_id=user_id, created_at=str(int(time.time() * 1000)))
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return db_conversation