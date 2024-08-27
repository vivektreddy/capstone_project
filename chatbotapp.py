from langchain.vectorstores import FAISS
import gradio as gr
from helper_functions import get_listing_id_from_url, get_url, response_list_check,\
remove_walking_times, truncate_before_second_occurrence, remove_repeated_sections
from helper_llm_functions import run_llm,summarize_reviews, listing_response
from langchain.llms import OpenAI
#from langchain.embeddings import BedrockEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import cohere
import os
import requests
import validators
import re
from pprint import pprint
import pandas as pd


file_paths = ['los angeles/reviews_detailed.csv','new york/reviews_detailed.csv','washington dc/reviews_detailed.csv','san francisco/reviews_detailed.csv']
review_dataframes = [pd.read_csv(file) for file in file_paths]
combined_review_data = pd.concat(review_dataframes, ignore_index=True)
 




os.environ["OPENAI_API_KEY"] = "<your openai api key here>"
llm = OpenAI()
emebeddings = OpenAIEmbeddings()
langchain_embedder = HypotheticalDocumentEmbedder.from_llm(llm, emebeddings, "web_search")
#each index file has a .faiss and .pkl
vector_store = FAISS.load_local("combined/combined_index_HyDE_embeddings",\
                                embeddings=langchain_embedder,\
                                allow_dangerous_deserialization=True)





#later add history object
def chatbot(query, type, history):
    
    retrieved_docs = vector_store.similarity_search(query, k=10)
    context = ' '.join([doc.page_content for doc in retrieved_docs])

    response = listing_response(query=query, context=context)
 
    #print('listing response: ',response)

    #\d+ 1 or more digits from 0-9, \. a literature '.', \s any whitespace character, \*\* two literal asterisks
    bullet_points = re.split(r'\n(?=\d+\.\s)', response)   
    bullet_points = [point.strip() for point in bullet_points if point.strip()]

    print('last bullet point: ',bullet_points[-1])  #you can split the last one on note: and scrap the last one
    print('len bullet points/# of retrieved listings: ', len(bullet_points))    

    response_list = [point for point in bullet_points if 'https://www.airbnb.com/rooms/' in point]
    print('# of retrieved listings with URLs: ',len(response_list))

    #if no listings (or no listings with URLs), return
    if not response_list:
        return "No properties match your criteria"
    else:
        updated_response_list = response_list_check(response_list)
        # if none remain. return
        if not updated_response_list:
            return "No current properties match your criteria"
        else:
            print('Properties exist with validated URLs')
            print('updated response list', updated_response_list)
            response_summary_dict = {}
            # if properties do currently exist, then make sure the property's listing ID exists
            # in the review table so you can return a summary
            pair_count = 0
            for listing in updated_response_list:

                listing_id = get_listing_id_from_url(listing)
                listing_id_present_in_review_df = listing_id in combined_review_data['listing_id'].values

                if listing_id_present_in_review_df:
                    
                    #key = listing_id value = listing llm output + summary llm output 
                    listing_summary = summarize_reviews(combined_review_data,listing_id)
                    response_summary_dict[listing_id]  = listing + "\n" + listing_summary
                    print(f"summary created for {listing_id}")
                else:
                    #key = listing_id value = listing llm output
                    response_summary_dict[listing_id] = listing
                    print(f"Couldnt map summary for {listing_id}")
                 
            #Create a Numbered List for Output Using Iterator
            output = []
            iterator = 1
            for i,g in response_summary_dict.items():
                #print(i, g) 
                output.append(f"{iterator}. {g[2:].replace('\n', '  \n')}")
                iterator+=1
            print(output)
            output = "\n\n".join(output)
            #remove any note
            pattern = r'Note: ".*?"'
            cleaned_output = re.sub(pattern, '', output)
            #should remove any function definitions until end
            pattern = r'\ndef".*?"'
            cleaned_output = re.sub(pattern, '', output)
            #cleaned_output = remove_repeated_sections(cleaned_output, min_words=9)
            return cleaned_output 
        
         
with gr.Blocks() as demo:
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

demo.launch(share=True)