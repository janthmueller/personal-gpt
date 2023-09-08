from langchain.chains.question_answering import load_qa_chain

import argparse
import json
from db import DB
from handle_secret_keys import set_secret_keys
from langchain.llms import OpenAI
from langchain.embeddings import *
import os

parser = argparse.ArgumentParser(description="Question answering.")
parser.add_argument(
    "-dbp",
    "--db-persist-dir",
    type=str,
    default="./db",
    help="Directory where the database is persisted.",
)
parser.add_argument(
    "-c", "--collection-name", type=str, help="Name of the collection to use."
)

parser.add_argument(
    "-ec",
    "--embedding-class",
    type=str,
    default="HuggingFaceEmbeddings",
    help="Choose embedding class to use.",
)

parser.add_argument(
    "-ek",
    "--embedding-kwargs",
    type=json.loads,
    default='{"model_name": "all-MiniLM-L6-v2"}',
    help="Kwargs for the embedding class.",
)  # -ek  {\"model\":\"text-embedding-ada-002\"} for OpenAIEmbeddings in windows
# TODO: replace -ek later on with config file for each embedding class


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Loading DB from {args.db_persist_dir}.")
    if not os.path.isdir(args.db_persist_dir):
        raise ValueError(f"No DB found at {args.db_persist_dir}.")
    db = DB(args.db_persist_dir)
    set_secret_keys()
    embedder_cls_name = args.embedding_class
    embedder_kwargs = args.embedding_kwargs
    print(f"Loading embedder {embedder_cls_name} with kwargs {embedder_kwargs}.")
    embedder = globals()[embedder_cls_name](**embedder_kwargs)
    print(f"Loding collection {args.collection_name}.")

    collection = db.get_langchain_collection(
        name=args.collection_name, embedder=embedder
    )
    llm = OpenAI()
    chain = load_qa_chain(llm)

    while True:
        question = input("Question: ")
        if question == "exit":
            break
        docs = collection.similarity_search(question)
        # print(docs)
        response = chain.run(input_documents=docs, question=question)
        print("Answer: " + response)
