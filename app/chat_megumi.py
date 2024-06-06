from fastapi.security import OAuth2PasswordRequestForm
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from typing import Annotated
from sqlalchemy.orm import Session
import shutil
import time
import dotenv
import os
from . import configs, crud, models, schemas, database, security

dotenv.load_dotenv()

database.Base.metadata.create_all(bind=database.engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_current_user_from_token(request: Request):
    token = request.headers.get('Authorization')
    if token is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = token.replace('Bearer ', '')
    try:
        user = security.decode_token(token)
        if user['expire_at'] < time.time():
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user
    except:
        raise HTTPException(status_code=401, detail="Unauthorized")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

model = ChatOpenAI(
    model=os.environ.get("GPT_MODEL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=float(os.environ.get("TEMPERATURE")),
    )

client = OpenAI()

store = {}

class Message(BaseModel):
    content: str

def get_chat_with_history():
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configs.chat_prompt['megumi'],),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chat_chain = chat_prompt | model

    with_message_history = RunnableWithMessageHistory(
        chat_chain,
        get_session_history,
        input_messages_key="messages",
    )
    return with_message_history


def get_translate_chain():
    translate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configs.translate_prompt,),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    translate_chain = translate_prompt | model
    return translate_chain


translate_chain = get_translate_chain()
chat_with_history = get_chat_with_history()

@app.post("/api/chat")
async def chat(message: Message, req: Request, db: Session = Depends(get_db)):
    # Get the user's info from the token
    current_user = get_current_user_from_token(req)
    # Get the user's conversation history
    user_id = current_user['user_id']
    # Check if the user's conversation history exists in the store
    if user_id not in store:
        # If not, create a new ChatMessageHistory object and add it to the store
        store[user_id] = ChatMessageHistory()
        # Get the user's conversation history from the database
        conversations = crud.get_user_conversations(db=db, user_id=user_id, limit=configs.max_chat_history)
        # Add the user's conversation history to the store
        conversations.reverse()
        for con in conversations:
            if con.role == "user":
                store[user_id].add_message(HumanMessage(content=con.message))
            elif con.role == "assistant":
                store[user_id].add_message(AIMessage(content=con.message))
    # If the user's conversation history exists in the store, check if it's longer than the maximum length
    elif len(store[user_id].messages) > configs.max_chat_history:
        # If it is, remove the oldest messages from the store
        store[user_id].messages = store[user_id].messages[-configs.max_chat_history:]
    # Add the user's message to database
    crud.create_conversation(
        db=db,
        conversation={
            "message": message.content,
            "role": "user",
        },
        user_id=user_id
    )
    # Set the user_id in the config
    config = {"configurable": {"session_id": user_id}}
    print('history: ', store[user_id])
    # Get the bot's response
    chat_reply = chat_with_history.invoke({"messages": [HumanMessage(content=message.content)]}, config=config).content
    # Translate the bot's response
    translation = translate_chain.invoke({"messages": [HumanMessage(content=chat_reply)]}).content
    # Add the bot's response to database
    crud.create_conversation(
        db=db,
        conversation={
            "message": chat_reply,
            "translation": translation,
            "role": "assistant",
        },
        user_id=current_user['user_id']
    )
    return {'message': chat_reply, 'translation': translation}


@app.post("/api/stt")
async def stt(audio: Annotated[UploadFile, File()], req: Request):
    # Get the user's info from the token
    current_user = get_current_user_from_token(req)
    # Check if the audio file is in a supported format
    supported_formats = ['audio/flac', 'audio/m4a', 'audio/mp3', 'audio/mp4', 'audio/mpeg', 'audio/mpga', 'audio/oga', 'audio/ogg', 'audio/wav', 'audio/webm']
    if audio.content_type not in supported_formats:
        return {"error": "Unsupported file format. Please upload a valid audio file."}
    # Save the audio file locally
    file_location = f"{audio.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    try:
        # Transcribe the audio file
        file = open(file_location, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
        )
        file.close()
    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"error": str(e)}

    return {
        "transcription": transcription.text,
    }


@app.post("/api/users", response_model=schemas.User)
def create_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db),
):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@app.get("/api/users", response_model=list[schemas.User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/api/users/{user_id}", response_model=schemas.User)
def read_user(
    user_id: int,
    db: Session = Depends(get_db),
):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/api/conversations/{user_id}", response_model=schemas.Conversation)
def create_conversation_for_user(
    user_id: int,
    conversation: schemas.ConversationCreate,
    role: str,
    db: Session = Depends(get_db),
):
    return crud.create_conversation(db=db, conversation=conversation, user_id=user_id, role=role)


@app.get("/api/conversations/{user_id}", response_model=list[schemas.Conversation])
def get_user_conversation(
    user_id: int,
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
):
    conversations = crud.get_user_conversations(db=db, user_id=user_id, skip=skip, limit=limit)
    conversations.reverse()
    return conversations


@app.get("/api/conversations", response_model=list[schemas.Conversation])
def read_conversations(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    conversations = crud.get_conversations(db, skip=skip, limit=limit)
    return conversations


@app.post("/api/token")
def create_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    if not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    token = security.generate_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": int(time.time()) + 60 * int(os.environ.get('TOKEN_EXPIRE_MINUTES')),
        "user_id": user.id,
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


