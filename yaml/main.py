from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import cosine_module 
import warnings

import json

app = FastAPI()
cosine_sim = cosine_module.CosineSimilarity()
warnings.filterwarnings("ignore", category=FutureWarning)

@app.post("/similitud-cosenos")
def rag_paradigm(texto1: str, texto2: str):
    result = ""
    try:
        result = cosine_sim.get_cosine_similarity(texto1, texto2)
    except Exception as e:
        print(f"Error: {e}")
    
    j1 = {
        "texto 1": texto1,
        "texto 2": texto2,
        "respuesta": result
    }
    jj = json.dumps(str(j1))

    return jj
