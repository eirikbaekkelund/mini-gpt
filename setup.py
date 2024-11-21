import os
import tiktoken
import numpy as np
from client import WeaviateClient
from typing import Tuple

def get_train_val_data(split: float=0.95) -> Tuple[str, str]:

    data_path = os.path.join(os.path.dirname(__file__), 'analytics.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    n = len(text)
    text_train = text[:int(n*split)]
    text_val = text[int(n*split):]

    return text_train, text_val

def export_to_binary(ids: np.ndarray, file_name: str):
    
    id_array = np.array(ids, dtype=np.uint16)
    id_array.tofile(os.path.join(os.path.dirname(__file__), f'{file_name}.bin'))

def tokenize(text: str, model: str = "gpt2"):
    encoder = tiktoken.get_encoding(model)
    ids = encoder.encode_ordinary(text)

    return ids


def doc_to_text(client: WeaviateClient, doc_id: int) -> str:
    chunks = client.get_chunks_of_document(doc_id=doc_id)
    if not chunks:
        return ""

    chunks = sorted(chunks, key=lambda x: (x.page, x.order))

    return "\n".join([chunk.text for chunk in chunks])

def write_to_txt(file: str="analytics"):
    client_params = dict(
        host = os.getenv('HOST'),
        port = os.getnenv('PORT'),
        grpc_host = os.getenv('GRPC_HOST'),
        grpc_port = os.getenv('GRPC_PORT'),
    )

    client = WeaviateClient(**client_params)
    client.connect()
    
    document_collection = client.get_collection_by_name("Document")
    doc_ids = [d.properties["doc_id"] for d in document_collection.iterator()]
    current_dir = os.path.dirname(__file__)
    with open(f"{current_dir}/{file}.txt", "w", encoding='utf-8') as f:
        for doc_id in doc_ids:
            text = doc_to_text(client, doc_id)
            if text:
                f.write(text)
                f.write("\n\n")
    
    client.close()


if __name__ == "__main__":
    
    #NOTE: uncomment to write the analytics text to a file
    # write_to_txt(client)
    text_train, text_val = get_train_val_data()
    train_ids, val_ids = tokenize(text_train), tokenize(text_val)
    print(f"Train: {len(train_ids)} tokens, Val: {len(val_ids)} tokens")
    export_to_binary(train_ids, "train")
    export_to_binary(val_ids, "val")
