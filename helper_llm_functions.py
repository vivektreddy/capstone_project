
"""from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

from langchain.vectorstores import FAISS
import gradio as gr
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import cohere
import os
import requests """
from langchain.llms.bedrock import Bedrock
from helper_functions import remove_walking_times
import re
import pandas as pd



def run_llm(prompt, temperature=0.8,max_gen_len=None):
    if max_gen_len is not None:
        print('here in max gen len')
        llm = Bedrock(credentials_profile_name = 'default',
                  #model_id="meta.llama3-8b-instruct-v1:0",
                  model_id="meta.llama3-1-8b-instruct-v1:0",
                  model_kwargs={
        "max_gen_len":max_gen_len,
        "temperature": temperature,
        "top_p": 0.9,
        })
    
    else:
      llm = Bedrock(credentials_profile_name = 'default',
                  #model_id="meta.llama3-8b-instruct-v1:0",
                  model_id="meta.llama3-1-8b-instruct-v1:0",
                  model_kwargs={
        #"max_gen_len":2000,
        "temperature": temperature,
        "top_p": 0.9,
        })
    response = llm.generate([prompt])

    response = response.generations[0][0].text
    pattern = r'Note:.*'
    response = re.sub(pattern, '',response, flags=re.DOTALL)
    return response

def summarize_reviews(df,listing_id,truncate_after_period=True):

  listing_reviews = df[df['listing_id'] == listing_id]

  if listing_reviews.empty:
      return "No reviews found for this listing."
  
  print('# listing_reviews: ',len(listing_reviews))
  if len(listing_reviews)<3:
     return f'Only {len(listing_reviews)} reviews.  Need at least 3 reviews to generate summary.'
  
  all_comments = " ".join(listing_reviews['comments'].dropna())
  print('length comments: ',len(all_comments))
  
  if len(all_comments)<1000:
      print('here in less than 1000: ',all_comments)
      prompt =  f""" 
        Provide a 3rd person plural concise summary of the following reviews in no more than 3 sentences:
        {all_comments}
      Summary:
      """
      response = run_llm(prompt,temperature=0,max_gen_len=100)  
  
  else:
      prompt =  f""" 
        Provide a concise overall summary of the following reviews.  It should be no more than 5 sentences.
          Do not output any code.  
      Reviews: {all_comments[:2200]}
      Summary:
      """
      response = run_llm(prompt,temperature=0,max_gen_len=200)  
 
  pattern = r'""".*'
  response = re.sub(pattern, '',response, flags=re.DOTALL)
    
  pattern = r"'''.*"
  response = re.sub(pattern, '',response, flags=re.DOTALL)
  
  pattern = r"```.*"
  response = re.sub(pattern, '',response, flags=re.DOTALL)


  if truncate_after_period:
    # If a period is found, slice the text up to that period
    last_period_index = response.rfind('.')
    if last_period_index != -1:
      response = response[:last_period_index + 1]

  return response 

def listing_response(query,context):

    prompt = "You are a helpful assistant. " + \
    "Return exactly what matches the conditions of the question from the context." + \
         "If nothing matches, return 'No properties match your criteria." + \
    "Provide results in an organized numbered list." + \
         "Don't return anything except for the list. Be brief and concise." + \
    "Include the listing URL and guest review in the answer. \n" + \
    f"Context: {context}\n " + \
    f"Question: {query}.  don't output any results above $300.\n " + \
    "Answer:"

    prompt = remove_walking_times(prompt)
    #print('prompt: ', prompt)
    response = run_llm(prompt,temperature=0.9)  
    
    pattern = r'Note:.*'
    response = re.sub(pattern, '',response, flags=re.DOTALL)

    pattern = r'#*'
    response = re.sub(pattern, '',response, flags=re.DOTALL)

    return response #


def precision_k(query,context):

    prompt = "How many elements in the context match " + \
    "all the criteria the question is asking for?  Pay attention to number of bedrooms, price, and location. "  + \
    f"Context: {context}\n " + \
    f"Question: {query}\n " + \
    f"Specify answer in this format: " + \
    "Precision: <Number of matching elements>/<Number of elements in the context> \n" + \
    "Number Elements Total: <# of elements in the context>. \n"

    response = run_llm(prompt,temperature=0,max_gen_len=20)  
    
    return response #
