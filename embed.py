from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    PythonLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
import os
import argparse
from db import DB
from handle_secret_keys import set_secret_keys
from langchain.embeddings import *
import json


file_types = ["txt", "pdf", "py"]
loader_types = [TextLoader, PyPDFLoader, PythonLoader]
text_splitter = [
    RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    RecursiveCharacterTextSplitter.from_language(
        chunk_size=1000, chunk_overlap=100, language=Language.PYTHON
    ),
]

parser = argparse.ArgumentParser(description="DB and collection management.")
parser.add_argument(
    "-p",
    "--document-path",
    type=str,
    default="./data",
    help="Path to the document/s or file to load.",
)
parser.add_argument(
    "-r",
    "--recursive",
    action="store_true",
    help="Recursively load files in subdirectories. Not necessary if path is a file.",
)
parser.add_argument(
    "-txt",
    "--txt",
    action="store_true",
    help="Load text file/s. Not necessary if path is a file.",
)
parser.add_argument(
    "-pdf",
    "--pdf",
    action="store_true",
    help="Load pdf file/s. Not necessary if path is a file.",
)
parser.add_argument(
    "-py",
    "--py",
    action="store_true",
    help="Load python file/s. Not necessary if path is a file.",
)

# do i need these?

parser.set_defaults(recursive=False)
parser.set_defaults(text=False)
parser.set_defaults(pdf=False)
parser.set_defaults(python=False)

###

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


def check_path_type(path):
    if os.path.isdir(path):
        return "dir"
    elif os.path.isfile(path):
        return "file"
    else:
        raise ValueError(f"{path} is not a valid path.")


def get_chunks(path, r, file_types, file_types_bool, loader_types, text_splitter):
    path_type = check_path_type(path)
    file_types_to_load = [
        file_types[i] for i in range(len(file_types)) if file_types_bool[i]
    ]
    loader_types_to_load = [
        loader_types[i] for i in range(len(loader_types)) if file_types_bool[i]
    ]
    if path_type == "file":
        if len(file_types_to_load) > 1:
            raise ValueError(
                f"Cannot load multiple file types from a single file.")
        elif not file_types_to_load:
            file_type = path.split(".")[-1]
            print(f"Detect file type: Determined file type to be {file_type}.")
            if file_type not in file_types:
                raise ValueError(f"File type {file_type} not supported.")
        else:
            file_type = file_types_to_load[0]

        loader_type = loader_types[file_types.index(file_type)]
        print(f"Matched file type {file_type} to loader {loader_type}.")
        print(f"Loading file {path}.")
        loader = loader_type(path)
        docs = loader.load_and_split()
        chunks = text_splitter[file_types.index(
            file_type)].split_documents(docs)
    else:
        if not file_types_to_load:
            raise ValueError(f"Must specify at least one file type to load.")

        chunks = []
        for file_type, loader_type in zip(file_types_to_load, loader_types_to_load):
            print(f"Matched file type {file_type} to loader {loader_type}.")
            print(f"Loading files of type {file_type} from directory {path}.")
            loader_cls = loader_type

            if r:
                glob = f"**/*.{file_type}"
            else:
                glob = f"*.{file_type}"
            docs = DirectoryLoader(
                path, glob=glob, loader_cls=loader_cls, recursive=r
            ).load_and_split()
            chunks += text_splitter[file_types.index(
                file_type)].split_documents(docs)

    return chunks


if __name__ == "__main__":
    args = parser.parse_args()

    path = args.document_path
    r = args.recursive
    file_types_bool = [args.txt, args.pdf, args.py]
    chunks = get_chunks(
        path, r, file_types, file_types_bool, loader_types, text_splitter
    )
    print(f"Loading DB from {args.db_persist_dir}.")
    if not os.path.isdir(args.db_persist_dir):
        raise ValueError(f"No DB found at {args.db_persist_dir}.")
    db = DB(args.db_persist_dir)
    print(f"Loding collection {args.collection_name}.")
    set_secret_keys()
    embedder_cls_name = args.embedding_class
    embedder_kwargs = args.embedding_kwargs

    print(
        f"Loading embedder {embedder_cls_name} with kwargs {embedder_kwargs}.")
    embedder = globals()[embedder_cls_name](**embedder_kwargs)
    print(f"Embedding {len(chunks)} chunks...")
    collection = db.get_langchain_collection(
        name=args.collection_name, embedder=embedder
    )
    collection.add_documents(chunks)
    print(
        f"Added {len(chunks)} embedded chunks to collection {args.collection_name}.")
    print(
        f"Collection {args.collection_name} contains {collection._collection.count()} elements."
    )

# TODO: rename private keys to secret keys
# TODO: add query and chatbot
