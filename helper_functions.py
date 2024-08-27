import requests
import validators
import re, os
import pandas as pd

def get_url(listing):
    """
    extract the url from the listing text
    """
    url_pattern = re.compile(r'(https://www.airbnb.com/rooms/\d+)')
    match = url_pattern.search(listing)
    url = match.group(0) 
    return url


def get_listing_id_from_url(place):
    """
    extract the listing id from the url
    """
    url = get_url(place)
    listing_id = url.split('/')[-1]
    return int(listing_id)

#with walking times included, input prompt could not process all 5 listings at once.  
# distances to places can still be inferred from reviews
def remove_walking_times(text):
    """
    original data had walking time to nearby landmarks in it for listings, 
    this removes it so input will be within the maximum context length to llm
    """
    # Define the regular expression pattern to match "Walking time to [X]: [number] minutes."
    pattern = r"Walking time to [^:]+: \d+(\.\d+)? minutes\.\n?"
    # Use re.sub to replace all occurrences of the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def truncate_before_second_occurrence(text, word):
    """
    in case input prompt is regenerated at beginning of output, 
    you can truncate parts you don't need based on 2nd occurence of word
    """
# Find the first occurrence
    first_position = text.find(word)
    if first_position != -1:
        # Find the second occurrence
        second_position = text.find(word, first_position + len(word))
        if second_position != -1:
            return text[second_position:]
    # Return the original string if the word is not found twice
    return text

def remove_repeated_sections(text, min_words=20):

    """
    Removes repeated sections of text that are at least min_words long.
    Returns the cleaned text and prints the deleted sections.
    """

    words = text.split()
    word_count = len(words)
    
    if word_count < min_words * 2:
        return text  # Not enough words for any repetition of min length

    seen = {}
    result = []
    deleted_sections = []
    i = 0
    
    while i < word_count:
        current_window = tuple(words[i:i + min_words])
        
        if current_window in seen:
            match_start = seen[current_window]
            match_len = min_words
            
            while (i + match_len < word_count and 
                   words[match_start + match_len] == words[i + match_len]):
                match_len += 1

            if match_len >= min_words:
                deleted_section = words[i:i + match_len]
                deleted_sections.append(' '.join(deleted_section))
                i += match_len  # Skip over the duplicated section
                continue
        else:
            seen[current_window] = i
        
        result.append(words[i])
        i += 1
    
    # Print deleted sections
    if deleted_sections:
        print("Deleted sections:")
        for section in deleted_sections:
            print(f"\n{section}\n")
    else:
        print("No duplicate section detected.")
    
    return ' '.join(result)


def response_list_check(listings):
    """
    Takes in a list of listings and returns a list of listings with valid URLs
    """
    updated_listings = []
    for listing in listings:
        url = get_url(listing)
        #validate URL
        if validators.url(url): 
            val_response = requests.head(url, allow_redirects=True, timeout=5) #head is like get request but only returns headers
            if val_response.status_code == 410:
                print(f"URL: {url} is gone status code 410")
                pass
            elif val_response.status_code == 200:
                print(f"URL: {url} is good status code 200")
                updated_listings.append(listing)
            else:
                print(f"Unexpected status code: {val_response.status_code}")
                print(f"URL: {url}")
    print('validated URLs')
    print(updated_listings)
    print('here')
    return updated_listings
