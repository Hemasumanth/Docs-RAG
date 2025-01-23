import os
import numpy as np
from pinecone import Pinecone
from pathlib import Path
from tqdm import tqdm
from sklearnrag.parser import extract_sections
from sklearnrag.chunk import chunk_sections
from sklearnrag.config import WORK_DIR
from pinecone import Pinecone
from sklearnrag.embedding import load_or_create_embeddings
from sklearnrag.vectordb import PineconeIndex


def build_index(docs_dir, chunk_size, chunk_overlap, embedding_model_name, embedding_dim):
    docs_dir = Path(WORK_DIR, "scikit-learn.org/stable/")
    html_files = [path for path in docs_dir.rglob("*html") if not path.is_dir() and "lite" not in path.parts]

    sections = list()

    for file in tqdm(html_files, desc="Extracting sections......"):
        for section in extract_sections({'path': str(file)}):
            sections.append(section)

    chunks = chunk_sections(sections,chunk_size, chunk_overlap)

    embedded_chunks = load_or_create_embeddings(embedding_model_name, chunks)

    index_name = f"{embedding_model_name.split('/')[-1]}-{chunk_size}"

    pc = PineconeIndex()

    existing_indexes = [    
        index_info["name"] for index_info in Pinecone().list_indexes()
    ]       

    if index_name not in existing_indexes:
        index = pc.create_index(index_name, embedding_dim)
        index = pc.get_index(index_name)
        pc.upsert_data(index, embedded_chunks)

    return index


def load_index(embedding_model_name, embedding_dim, chunk_size, chunk_overlap, docs_dir, index_name):
    index_name = f"{embedding_model_name.split('/')[-1]}-{chunk_size}".lower()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing_indexes = [    
        index_info["name"] for index_info in pc.list_indexes()
    ]   

    if index_name in existing_indexes:
        index = pc.Index(index_name)
    else:
        index = build_index(docs_dir, chunk_size, chunk_overlap, embedding_model_name, embedding_dim)
    return index


def semantic_search(query, index, embedding_model, k = 5):
    embedding = np.array(embedding_model.embed_query(query))

    result = index.query(
    vector=embedding.tolist(),
    top_k = k if k else 1,
    include_values=True,
    include_metadata=True
    )
    semantic_context = [{"id": row['id'],
                        "text": row['metadata']['text'],
                        "source": row['metadata']['source']} for row in result['matches']][:k]
    
    return semantic_context