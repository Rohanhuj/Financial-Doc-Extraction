import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import sqlite3
import re
import json
import time
#from tqdm.notebook import tqdm
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional, TypedDict
from getpass import getpass
from sentence_transformers import CrossEncoder

# Data Ingestion & Processing
#from unstructured.partition.html import partition_html
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
#from unstructured.documents.elements import element_from_dict
#from unstructured.partition.utils.element_utils import element_from_dict
from unstructured.staging.base import elements_from_dicts

# LLMs and Embeddings
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.callbacks.base import BaseCallbackHandler

# Vector Store
import qdrant_client
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
#from fastembed.embedding import TextEmbedding
from fastembed.sparse import SparseTextEmbedding

# Agent & Graph Components
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
import graphviz

print("All libraries imported successfully!")


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")

if "LANGCHAIN_API_KEY" not in os.environ:
    if "LANGSMITH_API_KEY" in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    else:
        os.environ["LANGCHAIN_API_KEY"] = getpass("Enter your LangSmith API Key: ")

print("API keys and environment variables are set.")
# Verify one of the keys to make sure it's loaded (we'll only show the first few chars for security)
print(f"OpenAI Key loaded: {os.environ['OPENAI_API_KEY'][:5]}...")

INPUT_DIR = f"input"
INPUT_DATAFILE_NAME = "NVIDIAAn.pdf"
INPUT_DATAFILE_PATH = os.path.join(INPUT_DIR, INPUT_DATAFILE_NAME)
JSON_SCHEMA_FILE_NAME = "schema_extraction.json"
OUTPUT_DIR = "output"
OUTPUT_JSON_FILE_NAME = "extracted.json"
OUTPUT_JSON_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_JSON_FILE_NAME)

COLLECTION_NAME_PREFIX = "fin_docs"
QDRANT_PATH = "./qdrant_db"
ENRICHED_CHUNKS_PATH = 'enriched_chunks.json'

class CustomException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class Config:
    def __init__(self):
        # Hard-code defaults here
        self.config = {}

    def read_config_file(self, config_file_name: str = "config.json"):
        try:
            with open(config_file_name) as config_file:
                self.config = json.loads(config_file.read())
            return self.config
        except Exception as e:
            print(f"Error was detected while reading {config_file_name}: {str(e)}. Default values applied")

class Json_schema:
    def __init__(self):
        # Hard-code defaults here
        self.json_schema = {}

    def read_json_schema_file(self, json_schema_file_name: str = "schema.json"):
        try:
            with open(json_schema_file_name) as json_schema_file:
                self.json_schema = json.loads(json_schema_file.read())
            return self.json_schema
        except Exception as e:
            print(f"Error was detected while reading {json_schema_file_name}: {str(e)}. Default values applied")

def extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses a JSON object from a string that may contain surrounding text
    or markdown code fences (e.g., ```json ... ```).

    Args:
        text (str): The input string containing the potential JSON data.

    Returns:
        Optional[Dict[str, Any]]: A Python dictionary if valid JSON is found and
                                  successfully parsed, otherwise None.
    """
    pattern = r"```(?:[a-zA-Z]+)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        raw_json = match.group(1)
    else:
        raw_json = text

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return None

# TODO : Implement common logger class to replace use of prints
