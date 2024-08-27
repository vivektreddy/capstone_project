from langchain.vectorstores import FAISS
import gradio as gr
from helper_functions import get_listing_id_from_url, get_url, remove_walking_times, truncate_before_second_occurrence 
from helper_llm_functions import run_llm,summarize_reviews, precision_k
from langchain.llms import OpenAI
#from langchain.embeddings import BedrockEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import cohere
import os
import requests
import validators
import re
import pandas as pd



os.environ["OPENAI_API_KEY"] = "<replace with your openai api key"
llm = OpenAI()
emebeddings = OpenAIEmbeddings()
langchain_embedder = HypotheticalDocumentEmbedder.from_llm(llm, emebeddings, "web_search")
#each index file has a .faiss and .pkl
vector_store = FAISS.load_local("combined/combined_index_HyDE_embeddings",
                                embeddings=langchain_embedder,
                                allow_dangerous_deserialization=True)





#later add history object
def chatbot(query, type, history):
    
    retrieved_docs = vector_store.similarity_search(query, k=5)
    print("retrieved docs",len(retrieved_docs))
    #context = ' '.join([doc.page_content for doc in retrieved_docs])
    context = [remove_walking_times(doc.page_content) for doc in retrieved_docs]
    print("context", len(context))
    print("walk time filtered context",(context))


    return precision_k(query=query, context=context)
  
  
precisions = []
#generate possible prompts many with chatgpt
prompts = ["Show me 1 bedroom places in sf under $200 near the swimming pool.",
           "Show me 1 bedroom places in new york under $300 with at least 4 stars.",
           "Show me 1 bedroom places in los angeles under $200 with an oven."]
for prompt in prompts:
    ret = (chatbot(prompt, None, None))
    match = re.search(r'Precision: (\d+)/(\d+)', ret)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        print(numerator, denominator)
        precision = numerator / denominator
        precisions.append(precision)
        print(precision)
print(sum(precisions)/len(precisions))

#use LLM as a judge
#how many of the results in the returned context are exact
      
print('end')
"""with gr.Blocks() as demo:
    gr.Markdown("# 🏠 Airbnb Chatbot Olivia")
    gr.Markdown("#### Ask a question regarding properties and specific features to get relevant recommendations")

    with gr.Tab("Query"):
        with gr.Column():
            response_output = gr.Markdown(elem_classes="gr-markdown")
            query_input = gr.Textbox(label="Query", placeholder="Enter your query here...", elem_classes="gr-textbox")
            submit_button = gr.Button("Submit", elem_classes="gr-button")
        
        submit_button.click(fn=chatbot, inputs=[query_input], outputs=response_output)

    with gr.Tab("About"):
        gr.Markdown("### About This Chatbot")
        gr.Markdown("This chatbot helps you find Airbnb listings based on your specific queries. It uses advanced language models to provide detailed summaries and analyses of listings and reviews.")

demo.launch(share=True)"""