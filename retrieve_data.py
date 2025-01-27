from google.cloud import bigquery
import pandas as pd
import os
import json


  
def get_reviews_from_db():
    proc_name = "GetReviews"
    results_set = get_procedure_results_set(proc_name)
    results_list = []
    for row in results_set:
        results_list.append({"id": row["id"], "text": row["text"]})  # Adjust the column name if needed

    df = pd.DataFrame(results_list)

    print(f"Stored procedure {proc_name} executed successfully.")

    return df


def get_procedure_results_set(proc_name):

    from utility.bigquery_config import setup_bigquery_config

    config = setup_bigquery_config()
    project_id = config["project_id"]
    dataset_id = config["dataset_id"]
    client = bigquery.Client(project=project_id)


    stored_proc = f"`{project_id}.{dataset_id}.{proc_name}`"

    query_job = client.query(f"CALL {stored_proc}()")

    results_set = query_job.result()

    return  results_set




def get_overall_sentiment_distribution():
    proc_name = "GetOverallSentimentDistribution"
    results_set = get_procedure_results_set(proc_name)
    
    results_list = []
    for row in results_set:
        results_list.append({"sentiment": row["sentiment"], "value": row["value"]})  # Adjust the column name if needed

    sentiment_distribution = pd.DataFrame(results_list)

    print(f"Stored procedure {proc_name} executed successfully.")
    
    return sentiment_distribution


def get_overall_sentiment_trends_data():
    proc_name = "GetOverallSentimentTrendsData"
    results_set = get_procedure_results_set(proc_name)
    
    results_list = []
    for row in results_set:
        results_list.append({"id": row["id"], "timestamp": row["timestamp"], "sentiment": row["sentiment"]})  # Adjust the column name if needed

    sentiment_trends = pd.DataFrame(results_list)

    print(f"Stored procedure {proc_name} executed successfully.")
    
    return sentiment_trends


def get_reviews_by_sentiment(input_sentiment):
    from utility.bigquery_config import setup_bigquery_config
    config = setup_bigquery_config()
    project_id = config["project_id"]
    dataset_id = config["dataset_id"]
    client = bigquery.Client(project=project_id)
    
    proc_name = "GetReviewsBySentiment"

    stored_proc = f"`{project_id}.{dataset_id}.{proc_name}`"

    query_job = client.query(f"CALL {stored_proc}({input_sentiment})")

    results_set = query_job.result()
    
    results_list = []
    for row in results_set:
        results_list.append({"id": row["id"], "text": row["text"]})  # Adjust the column name if needed

    sliced_reviews = pd.DataFrame(results_list)

    print(f"Stored procedure {proc_name} executed successfully.")
    
    return sliced_reviews