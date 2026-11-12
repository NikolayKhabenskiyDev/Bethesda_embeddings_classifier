import numpy as np
import pandas as pd
import re
import random as rd
import sentence_transformers as st
from pathlib import Path

def create_training_examples(clean_data):
    max_pairs = 2000
    examples = []

    category_to_indices = {'I': [], 'II': [], 'III': [], 'IV': [], 'V': [], 'VI': []}
    for idx, cat in enumerate(clean_data['Сategory']):
            category_to_indices[cat].append(idx)

    categories = list(category_to_indices.keys())
    pairs_per_category = max_pairs // len(categories)
    examples_per_category = pairs_per_category // 6
    for cat in categories:
        indices = category_to_indices[cat]
        for ii in range(min(len(indices), examples_per_category)):
            for _ in range(3):
                jj = rd.choice(indices)
                examples.append(st.InputExample(
                    texts=[
                        clean_data.iloc[indices[ii]]['Description'],
                        clean_data.iloc[jj]['Description']
                          ],
                    label=1))

        other_cat = rd.choice([cat2 for cat2 in categories if cat2 != cat])

        for ii in range(min(len(indices), examples_per_category)):
            for _ in range(3):
                jj = rd.choice(category_to_indices[other_cat])
                examples.append(st.InputExample(
                    texts=[
                        clean_data.iloc[indices[ii]]['Description'],
                        clean_data.iloc[jj]['Description']
                          ],
                    label=0))
    return examples

train_examples = create_training_examples(clean_data)
model = st.SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
train_dataloader = tud.DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = st.losses.ContrastiveLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    optimizer_params={'lr': 2e-5},
    output_path='./bethesda_medical_model',
    save_best_model=True,
    show_progress_bar=True
)

model = st.SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(clean_data['Description'].tolist(),
                         convert_to_tensor=False,
                        show_progress_bar=True)
trained_model = st.SentenceTransformer('./bethesda_medical_model')
embeddings = trained_model.encode(
    clean_data['Description'].tolist(),
    convert_to_tensor=False,
    show_progress_bar=True,
    normalize_embeddings=True,
    batch_size=32
)
embeddings_df = pd.DataFrame(embeddings)