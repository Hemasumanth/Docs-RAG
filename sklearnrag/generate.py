import json
import pickle
import re
import time
from pathlib import Path
import torch

from IPython.display import JSON, clear_output, display
from tqdm import tqdm
from pinecone import Pinecone
import os

from sklearnrag.config import WORK_DIR
from sklearnrag.embedding import get_embedding_model
from sklearnrag.search import load_index, semantic_search
from sklearnrag.utils import get_client, get_num_tokens, trim


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def response_stream(chat_completion):
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def prepare_response(chat_completion, stream):
    if stream:
        return response_stream(chat_completion)
    else:
        return chat_completion.choices[0].message.content


def send_request(
    llm,
    messages,
    max_tokens=None,
    temperature=0.0,
    stream=False,
    max_retries=1,
    retry_interval=60,
):
    retry_count = 0
    client = get_client(llm=llm)
    while retry_count <= max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=llm,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                messages=messages,
            )
            return prepare_response(chat_completion, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""


def generate_response(
    llm,
    max_tokens=None,
    temperature=0.0,
    stream=False,
    system_content="",
    assistant_content="",
    user_content="",
    max_retries=1,
    retry_interval=60,
):
    """Generate response from an LLM."""
    messages = [
        {"role": role, "content": content}
        for role, content in [
            ("system", system_content),
            ("assistant", assistant_content),
            ("user", user_content),
        ]
        if content
    ]
    return send_request(llm, messages, max_tokens, temperature, stream, max_retries, retry_interval)


class QueryAgent:
    def __init__(
        self,
        embedding_model_name="thenlper/gte-large",
        index=pc.Index("gte-large-750"),
        llm="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.0,
        max_context_length=32768,
        system_content="",
        assistant_content=""
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": 100}
        )

        self.index = index

        # LLM
        max_context_length = 4096 if llm == "gpt-4-turbo-2024-04-09" else max_context_length
        self.llm = llm
        self.temperature = temperature
        self.context_length = int(
            0.5 * max_context_length
        ) - get_num_tokens(  # 50% of total context reserved for input
            system_content + assistant_content
        )
        self.max_tokens = int(
            0.5 * max_context_length
        )  # max sampled output (the other 50% of total context)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(
        self,
        query,
        num_chunks=5,
        stream=True
    ):
        # Get top_k context
        context_results = semantic_search(
            query=query, embedding_model=self.embedding_model, index = self.index, k=num_chunks
        )

        # Generate response
        document_ids = [item["id"] for item in context_results]
        context = [item["text"] for item in context_results]
        sources = set([item["source"] for item in context_results])
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=trim(user_content, self.context_length)
        )

        # Result
        result = {
            "question": query,
            "sources": sources,
            "document_ids": document_ids,
            "answer": answer,
            "llm": self.llm
        }
        return result


# Generate responses
def generate_responses(
    experiment_name,
    chunk_size,
    num_chunks,
    embedding_model_name,
    embedding_dim,
    llm,
    temperature,
    max_context_length,
    system_content,
    assistant_content,
    docs_dir,
    experiments_dir,
    references_fp,
    chunk_overlap = 100,
    num_samples=None
):
    # Build index
    index = load_index(
        embedding_model_name=embedding_model_name,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        docs_dir=docs_dir,
        index_name = f"""{embedding_model_name.split("/")[-1]}-{chunk_size}"""
    )

    # Query agent
    agent = QueryAgent(
        embedding_model_name=embedding_model_name,
        index=index,
        llm=llm,
        temperature=temperature,
        system_content=system_content,
        assistant_content=assistant_content,
    )

    # Generate responses
    results = []
    with open(Path(references_fp), "r") as f:
        questions = [item["question"] for item in json.load(f)][:num_samples]
    for query in tqdm(questions):
        result = agent(query=query,
                    num_chunks=num_chunks,
                    stream=False)
        results.append(result)
        clear_output(wait=True)
        display(JSON(json.dumps(result, indent=2, default=set_default)))


    # Save to file
    responses_fp = Path(WORK_DIR, experiments_dir, "responses", f"{experiment_name}.json")
    responses_fp.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": experiment_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": num_chunks,
        "embedding_model_name": embedding_model_name,
        "llm": llm,
        "temperature": temperature,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
        "docs_dir": str(docs_dir),
        "experiments_dir": str(experiments_dir),
        "references_fp": str(references_fp),
        "num_samples": len(questions)
    }
    responses = {
        "config": config,
        "results": results
    }

    # Convert any sets in 'responses' to lists
    for key, value in responses.items():
        if isinstance(value, set):
            responses[key] = list(value)

    with open(responses_fp, "w") as fp:
        json.dump(responses, fp, indent=4, default=set_default)