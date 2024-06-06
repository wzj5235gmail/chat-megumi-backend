from pydantic import BaseModel


class ConversationBase(BaseModel):
    message: str
    role: str
    translation: str | None = None

class ConversationCreate(ConversationBase):
    pass


class Conversation(ConversationBase):
    id: int
    user_id: int
    created_at: str

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    conversations: list[Conversation] = []

    class Config:
        orm_mode = True