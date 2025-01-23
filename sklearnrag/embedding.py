import os
import torch
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from pathlib import Path
from sklearnrag.config import WORK_DIR
import json


def load_or_create_embeddings(model_name, chunks):
    embeddings_file_name = f"{model_name.split('/')[-1]}-embedded-chunks.json"

    embeddings_file_path = Path(WORK_DIR, "saved_embeddings")
    embeddings_file_path.mkdir(parents=True, exist_ok=True)
    embeddings_file_path = Path(embeddings_file_path, embeddings_file_name)
    
    # Check if embeddings file exists
    if embeddings_file_path.exists():
        with open(embeddings_file_path, 'r') as file:
            embeddings = json.load(file)
    else:
        # Initialize the embedding model
        embedder = EmbedChunks(model_name=model_name)
        embeddings = embedder.process_chunks(chunks)
        
        # Save the embeddings to a file
        with open(embeddings_file_path, 'w') as file:
            json.dump(embeddings, file, indent=4)
    
    return embeddings


def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    if embedding_model_name == "text-embedding-3-large":
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    return embedding_model


class EmbedChunks:
    def __init__(self, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": 100},
        )

    def __call__(self, batch):
        texts = [chunk["text"] for chunk in batch]
        embeddings = self.embedding_model.embed_documents(texts)
        for i, chunk in enumerate(batch):
            chunk["embeddings"] = embeddings[i]
        return batch
    
    def process_chunks(self, chunks, batch_size=100):
        embedded_chunks = []
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]
            embedded_batch = self(batch) 
            embedded_chunks.extend(embedded_batch)
        return embedded_chunks