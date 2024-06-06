import shutil
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile
from typing import Annotated
import dotenv
import os
import model_config

dotenv.load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ChatOpenAI(
    model=os.environ.get("GPT_MODEL"), 
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=float(os.environ.get("TEMPERATURE")),
    )
client = OpenAI()
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", model_config.chat_prompt['setsuna'],),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chat_chain = chat_prompt | model

with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="messages",
)

translate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", model_config.translate_prompt,),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

translate_chain = translate_prompt | model

class Message(BaseModel):
    content: str

@app.post("/api/translate")
async def translate(message: Message):
    return {
        'translation': translate_chain.invoke({"messages": [HumanMessage(content=message.content)]}).content
    }

@app.post("/api/chat")
async def chat(message: Message):
    config = {"configurable": {"session_id": "test"}}
    res = with_message_history.invoke(
            {"messages": [HumanMessage(content=message.content)]}, config=config
        )
    return {'message': res.content}

@app.post("/api/stt")
async def stt(audio: Annotated[UploadFile, File()]):
    # 打印上传文件的详细信息以便调试
    print(f"Received file: {audio.filename}")
    print(f"Content type: {audio.content_type}")
    supported_formats = ['audio/flac', 'audio/m4a', 'audio/mp3', 'audio/mp4', 'audio/mpeg', 'audio/mpga', 'audio/oga', 'audio/ogg', 'audio/wav', 'audio/webm']
    if audio.content_type not in supported_formats:
        return {"error": "Unsupported file format. Please upload a valid audio file."}

    file_location = f"{audio.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    try:
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

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile=os.environ.get("SSL_KEY"), ssl_certfile=os.environ.get("SSL_CERT"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
