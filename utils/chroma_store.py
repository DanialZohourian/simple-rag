from dataclasses import dataclass
from pathlib import Path
import chromadb

@dataclass
class ChromaStore:
    persist_dir: Path
    collection_name: str = "rag_chunks"

    def __post_init__(self):
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, *, file_id: str, file_name: str, chunks: list[dict], embeddings: list[list[float]]):
        # ids: file_id:chunk_number
        ids = [f"{file_id}:{c['chunk_number']}" for c in chunks]
        metadatas = []
        documents = []
        for c in chunks:
            metadatas.append({
                # required + extra internal file_id
                "file_id": file_id,
                "file_name": c["file_name"],
                "chunk_number": int(c["chunk_number"]),
                "page_number": str(c["page_number"]),
                "embedded_text": c["embedded_text"],
            })
            documents.append(c["embedded_text"])

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_file(self, *, file_id: str):
        # deletes all chunks for that file
        self.collection.delete(where={"file_id": file_id})

    def query(self, *, query_embedding: list[float], n_results: int = 6):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
