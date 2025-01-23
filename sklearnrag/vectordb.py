import os
import time
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


class PineconeIndex(Pinecone):
    def __init__(self):
        super().__init__(api_key=os.environ["PINECONE_API_KEY"])
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    def get_existing_indexes(self):
        return [index["name"] for index in self.pc.list_indexes()]
    
    def get_index(self, index_name):
        existing_indexes = self.get_existing_indexes()
        if index_name not in existing_indexes:
            raise ValueError(f"Index {index_name} does not exist.\n----------------------")
        
        return self.pc.Index(index_name)

    def create_index(self, index_name, dimension, metric="cosine"):
        existing_indexes = self.get_existing_indexes()
        if index_name in existing_indexes:
            raise Warning(f"Index {index_name} already exists.\n----------------------")
        
        self.pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud='aws',
                                region='us-west-2'
                                )
        )

        while not self.pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        print(f"Index {index_name} created.\n----------------------")
        return self.pc.Index(index_name)
    
    def delete_index(self, index_name):
        existing_indexes = self.get_existing_indexes()
        if index_name not in existing_indexes:
            raise ValueError(f"Index {index_name} does not exist to delete.")
        
        self.pc.delete_index(index_name)
        print(f"Index {index_name} deleted.\n----------------------")

    def upsert_data(self, index, embedding_chunks, batch_size=100):
        upsert_data = [
            (str(i), chunk["embeddings"], {"text": chunk["text"], "source": chunk["source"]})
            for i, chunk in enumerate(embedding_chunks)
        ]

        for i in tqdm(range(0, len(upsert_data), batch_size)):
            batch = upsert_data[i:i+batch_size]
            index.upsert(vectors=batch)
        
        while index.describe_index_stats()['total_vector_count'] != len(embedding_chunks):
             time.sleep(1)
        
        print(f"Successfully upserted the data!.")

















