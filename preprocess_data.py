# # Install spaCy 
# pip install spacy

# # Download spaCy's English language model
# python -m spacy download en_core_web_sm

import spacy
from retrieve_data import get_reviews_from_db
import pandas as pd
import spacy

from google.cloud import bigquery
from transformers import pipeline


from utility.bigquery_config import setup_bigquery_config

from concurrent.futures import ThreadPoolExecutor


def save_to_bigquery(df, table_id):
    try:
        client = bigquery.Client()
        job = client.load_table_from_dataframe(df, table_id)
        job.result()  
        print(f"Data uploaded to BigQuery table {table_id}.")
    except Exception as e:
        print(f"Error saving to BigQuery: {e}")


def summarize_texts(texts, max_length= 400, limit = 250):
    summarizer = pipeline("summarization", model="google/bigbird-pegasus-large-arxiv")
    summaries = []
    for text in texts:
        original_length = len(text.split())
        if original_length > limit:  # Check if review is longer than 2 words
            summarized = summarizer([text], max_length= min(max_length, original_length), truncation=True)
            summaries.append(summarized[0]['summary_text'])
            
        else:
            # Skip summarization for very short reviews
            summaries.append(text)
    
    return summaries





def preprocess_review_texts(texts):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
 
    clean_texts = []
    
    for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "parser"]):

        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha]

        clean_texts.append(" ".join(tokens))
    
    return clean_texts




def find_sentiment(model_name, texts, batch_size=8):
    if model_name == "BERT-pretrained":
        sentiment_pipeline = pipeline('sentiment-analysis')
        
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_results = sentiment_pipeline(batch)
                sentiments.extend([result['label'] for result in batch_results])
            except Exception as e:
                print(f"Error processing sentiment batch {i // batch_size}: {e}")
        return sentiments




if __name__ == '__main__':
    
    config = setup_bigquery_config()
    project_id = config["project_id"]
    dataset_id = config["dataset_id"]
    
    reviews_data = get_reviews_from_db()
    print("Data retrieved successfully.")  
    
    
    # Summarize the reviews
    summaries = summarize_texts(reviews_data['text'].tolist())
    print("summarization finished")
    
    
    # Preprocess the summarized text
    # reviews_data['processed_text'] = preprocess_review_texts(summaries)
    # print("preprocessing finished")
    
    
    # Perform sentiment analysis
    reviews_data['sentiment'] = find_sentiment("BERT-pretrained", summaries)
    print("Sentiment analysing finished")
    
    
    print(reviews_data.head())
    reviews_data = reviews_data.drop(columns=['text'])
    save_to_bigquery(reviews_data, f"{project_id}.{dataset_id}.Sentiments")
    
    
    
    
    
    
# def summarize_texts(texts, min_length=512, default_max_length=512, BIGBIRD_MAX_LENGTH = 4096):
#     # Use BigBird for summarization
#     summarizer = pipeline("summarization", model="google/bigbird-pegasus-large-arxiv", truncation=True)

#     def summarize_single_text(text):
#         # Ensure text is within the BigBird's max token limit
#         if len(text.split()) > BIGBIRD_MAX_LENGTH:
#             text = " ".join(text.split()[:BIGBIRD_MAX_LENGTH])  # Truncate to max length
#         if len(text.split()) > min_length:
#             return summarizer(
#                 text,
#                 max_length=default_max_length,  # Explicitly set max_length
#                 min_length=min_length // 2,
#                 do_sample=False,
#                 truncation=True  # Ensure truncation is applied
#             )[0]['summary_text']
#         return text

#     summarized_texts = []

#     with ThreadPoolExecutor() as executor:
#         # Process the texts in parallel
#         summarized_texts = list(executor.map(summarize_single_text, texts))

#     return summarized_texts

# # Summarize texts using BigBird (supports longer documents)
# def summarize_texts(texts, min_length=512, default_max_length=512):
#     # Use BigBird for summarization
#     summarizer = pipeline("summarization", model="google/bigbird-pegasus-large-arxiv", tokenizer=tokenizer, truncation=True)

#     def summarize_single_text(text):
#         # Tokenize and truncate the text if it's too long
#         tokens = tokenizer.encode(text, truncation=True, max_length=BIGBIRD_MAX_LENGTH, return_tensors="pt")
        
#         # If token length exceeds the limit, truncate it
#         if len(tokens[0]) > BIGBIRD_MAX_LENGTH:
#             tokens = tokens[:, :BIGBIRD_MAX_LENGTH]
        
#         # Decode and return the summary
#         summary = summarizer(tokens)
#         return summary[0]['summary_text']

#     summarized_texts = []

#     with ThreadPoolExecutor() as executor:
#         # Process the texts in parallel
#         summarized_texts = list(executor.map(summarize_single_text, texts))

#     return summarized_texts