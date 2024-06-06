from fastapi import Depends
import bcrypt
import jwt
import os
import dotenv
import time
from fastapi.security import OAuth2PasswordBearer

dotenv.load_dotenv()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
HASH_SECRET_KEY = os.environ.get('HASH_SECRET_KEY')

def hash_password(password):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'expire_at': int(time.time()) + 60 * int(os.environ.get('TOKEN_EXPIRE_MINUTES'))
    }
    token = jwt.encode(payload, HASH_SECRET_KEY, algorithm='HS256')
    return token

def decode_token(token):
    try:
        payload = jwt.decode(token, HASH_SECRET_KEY, algorithms=['HS256'])
        if payload['expire_at'] < int(time.time()):
            return None
        return payload
    except Exception as e:
        raise Exception(e)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    return user