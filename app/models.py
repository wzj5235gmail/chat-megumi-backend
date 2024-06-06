from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

# import database
from . import database


class User(database.Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)

    conversation = relationship("Conversation", back_populates="user")


class Conversation(database.Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    message = Column(Text)
    translation = Column(Text)
    created_at = Column(String(255), index=True)
    role = Column(String(255), index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="conversation")