# capstone_project
MIDS Capstone Project

1. data/combined_added_distances_and_url_df contains post processed data.  only final column is used for embedding.  you can further modify the data format prior to generating vector
2. run generate_and_save_vector_store making sure openai_api_key and correct data path is provided
3. data/dc_reviews_detailed, data/la_reviews_detailed, data/ny_reviews_detailed, data/sf_reviews_detailed contains reviews for each guest stay for those cities
4. run chatbotapp making sure paths are correct to each reviews file