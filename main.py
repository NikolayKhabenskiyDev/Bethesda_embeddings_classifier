import numpy as np
import pandas as pd
import re
import random as rd
import sentence_transformers as st
import torch.utils.data as tud
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

import prepare_data
import classifier

model = st.SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

if Path('data/Medical_embeddings.xlsx').exists():
    embeddings_df = pd.read_excel("data/Medical_embeddings.xlsx")
else:
    embeddings_df = prepare_data.create_embeddings("data/Описания_и_заключения_цитологии_ЭНЦ_ослепленные_2013_2023_1.xlsx", model,
                                                   "data/Prepared_data.xlsx", "data/Medical_embeddings.xlsx")

if Path('random_forest.joblib').exists():
    rfc = joblib.load('random_forest.joblib')
    pca = joblib.load('pca_model.joblib')
else:
    pca, rfc = classifier.create_random_forest("data/Medical_embeddings.xlsx")

print("success")
print("running app...")
class description(BaseModel):
    text: str | None = None

class output(BaseModel):
    embed: list | None = None
    categ: str | None = None

app = FastAPI()

@app.post("/analyse/")
async def create_embedding(descr: description):
    embedding = model.encode(
        [descr.text],
        convert_to_tensor=False,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=1
    )
    embedding_pca = pca.transform(embedding)
    category = rfc.predict(embedding_pca)

    out = output(embed=embedding.tolist(), categ=category[0])
    return out

uvicorn.run(app, host="0.0.0.0", port=8000)
print("success")