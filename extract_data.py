from datasets import load_dataset
from google.cloud import bigquery
import pandas as pd
import os
import json


def get_data(url, category):
    
    reviews_dataset = load_dataset(url, f"raw_review_{category}", trust_remote_code=True)
    products_dataset = load_dataset(url, f"raw_meta_{category}", split="full", trust_remote_code=True)
    return reviews_dataset, products_dataset


def get_and_store_required_data(url, category, project_id, dataset_id):

    reviews_dataset, products_dataset = get_data(url, category)
    
    reviews_data_frame = pd.DataFrame(reviews_dataset["full"].select_columns(["rating", "title", "text", "asin", "parent_asin", "helpful_vote", "timestamp", "verified_purchase"]))
    products_data_frame = pd.DataFrame(products_dataset.select_columns(["main_category", "title", "features", "price", "description", "parent_asin", "average_rating", "rating_number"]))


    save_to_bigquery(reviews_data_frame, f"{project_id}.{dataset_id}.reviews")
    save_to_bigquery(products_data_frame, f"{project_id}.{dataset_id}.products")



def save_to_bigquery(df, table_id):
    
    client = bigquery.Client()
    job = client.load_table_from_dataframe(df, table_id)
    job.result() 

    print(f"Data uploaded to BigQuery table {table_id}.")




from utility.bigquery_config import setup_bigquery_config

config = setup_bigquery_config()
project_id = config["project_id"]
dataset_id = config["dataset_id"]


url = "McAuley-Lab/Amazon-Reviews-2023"
category = "All_Beauty"

get_and_store_required_data(url, category, project_id, dataset_id)


