
import matplotlib.pyplot as plt
import seaborn as sns
from retrieve_data import get_overall_sentiment_distribution, get_overall_sentiment_trends_data, get_reviews_by_sentiment
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns
import os


def Calculate_Overall_Sentiment_Distribution(product_id=None):
    if product_id:
        sentiment_distribution = get_overall_sentiment_distribution(product_id)

    else:

        sentiment_distribution = get_overall_sentiment_distribution()

    output_dir = "charts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # PIE CHART
    plt.figure(figsize=(6, 6))
    sentiment_distribution['value'].plot.pie(
        autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral']
    )
    plt.title('Overall Sentiment Distribution')
    plt.ylabel('')

    pie_chart_path = os.path.join(
        output_dir, 'overall_sentiment_pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()

    # BAR CHART
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=sentiment_distribution.index,
        y=sentiment_distribution['value'],
        palette=['lightgreen', 'lightcoral']
    )
    plt.title('Overall Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')

    bar_chart_path = os.path.join(
        output_dir, 'overall_sentiment_bar_chart.png')
    plt.savefig(bar_chart_path)
    plt.close()

    return sentiment_distribution, pie_chart_path, bar_chart_path


def Calculate_overall_sentiment_trends_over_time():

    df = get_overall_sentiment_trends_data()
    sentiment_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)

    # Convert timestamp to datetime and extract monthly periods (timestamps are in milliseconds)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['month'] = df['date'].dt.to_period('M')

    # Group by month and calculate average sentiment score
    monthly_sentiment = df.groupby(
        'month')['sentiment_score'].mean().reset_index()

    # Convert the Period data to string for plotting
    monthly_sentiment['month'] = monthly_sentiment['month'].astype(str)

    print(monthly_sentiment)

    # Line graph
    plt.figure(figsize=(10, 6))
    plt.plot(
        monthly_sentiment['month'],
        monthly_sentiment['sentiment_score'],
        marker='o',
        linestyle='-',
        color='blue',
    )
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    # Create a pivot table for the heatmap
    df['year'] = df['date'].dt.year
    df['month_name'] = df['date'].dt.month_name()

    heatmap_data = df.pivot_table(
        index='month_name', columns='year', values='sentiment_score', aggfunc='mean'
    )

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Average Sentiment Score'},
    )
    plt.title('Sentiment Trends Heatmap')
    plt.xlabel('Year')
    plt.ylabel('Month')
    plt.show()


def preprocess_text(texts):
    nlp = spacy.load("en_core_web_sm")
    processed_texts = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        tokens = [token.lemma_.lower()
                  for token in doc if token.is_alpha and not token.is_stop]
        processed_texts.append(" ".join(tokens))
    return processed_texts


def extract_keywords(texts, top_n=10):
    nlp = spacy.load("en_core_web_sm")
    keywords = []
    for doc in nlp.pipe(texts):
        keywords.extend([token.lemma_.lower()
                        for token in doc if token.is_alpha and not token.is_stop])
    return Counter(keywords).most_common(top_n)


def get_common_themes_in_different_reviews():

    positive_reviews = get_reviews_by_sentiment("positive")
    negative_reviews = get_reviews_by_sentiment("negative")

    positive_reviews_processed = preprocess_text(
        positive_reviews['text'].to_list()).to_pandas()
    negative_reviews_processed = preprocess_text(
        negative_reviews['text'].to_list()).to_pandas()

    # Extract keywords for positive and negative reviews
    positive_keywords = extract_keywords(positive_reviews_processed, top_n=10)
    negative_keywords = extract_keywords(negative_reviews_processed, top_n=10)

    print("Positive Keywords:", positive_keywords)
    print("Negative Keywords:", negative_keywords)

    # Word cloud for positive reviews
    positive_wordcloud = WordCloud(background_color="white").generate(
        " ".join(positive_reviews_processed))
    plt.figure(figsize=(8, 6))
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Common Themes in Positive Reviews")
    plt.show()

    # Word cloud for negative reviews
    negative_wordcloud = WordCloud(background_color="white").generate(
        " ".join(negative_reviews_processed))
    plt.figure(figsize=(8, 6))
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Common Themes in Negative Reviews")
    plt.show()

    # Convert keywords to DataFrame
    positive_keywords_df = pd.DataFrame(
        positive_keywords, columns=["Keyword", "Frequency"])
    negative_keywords_df = pd.DataFrame(
        negative_keywords, columns=["Keyword", "Frequency"])

    # Bar chart for positive keywords
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Frequency", y="Keyword",
                data=positive_keywords_df, palette="viridis")
    plt.title("Top Keywords in Positive Reviews")
    plt.show()

    # Bar chart for negative keywords
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Frequency", y="Keyword",
                data=negative_keywords_df, palette="viridis")
    plt.title("Top Keywords in Negative Reviews")
    plt.show()


if __name__ == "__main__":

    # print(Calculate_Overall_Sentiment_Distribution('069267599X'))
    Calculate_overall_sentiment_trends_over_time()
