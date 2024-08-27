# -*- coding: utf-8 -*-
"""
HypotheticalDocumentEmbedder takes a default template or custom prompt.
Here the predefined web search template is used, which is the following.
this output is saved in the vector store
web_search_template = "Please write a passage to answer the question
Question: {QUESTION}
Passage:
"""

!pip install langchain faiss-gpu  langchain_community openai tiktoken  --quiet

import numpy as np
import pandas as pd
import os

#OPTIONAL: if saving and loading from google drive
from google.colab import drive
drive.mount('/content/drive')

from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

os.environ["OPENAI_API_KEY"] = "<replace with your key>"

## Create and Save Vector Stores

df = pd.read_csv('<replace with path to your data to be embedded and saved in vector store')
df = df.description_new
documents = df.astype(str).tolist()
print(len(documents))

#openai api key set as env variable
llm = OpenAI()
embeddings = OpenAIEmbeddings()


langchain_embedder = HypotheticalDocumentEmbedder.from_llm(llm, embeddings, "web_search")

#takes several minutes to create vector store for 4 cities
vector_store = FAISS.from_texts(texts=documents, embedding=langchain_embedder)

# change path
vector_store.save_local("<replace with save location>")

#
vector_store = FAISS.load_local("<replace with save location>",
                                embeddings=langchain_embedder,
                                allow_dangerous_deserialization=True)


