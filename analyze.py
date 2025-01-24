
import matplotlib.pyplot as plt
import seaborn as sns
from retrieve_data import get_sentiment_distribution, get_sentiment_trends_data
import pandas as pd

def Calculate_Sentiment_Distribution():
  

  sentiment_distribution = get_sentiment_distribution()
  
  plt.figure(figsize=(6, 6))
  sentiment_distribution.plot.pie(
      autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'gold']
  )
  plt.title('Overall Sentiment Distribution')
  plt.ylabel('')  # Removes the y-axis label
  plt.show()
  
  
  plt.figure(figsize=(8, 5))
  sns.barplot(
      x=sentiment_distribution.index,
      y=sentiment_distribution.values,
      palette=['lightgreen', 'lightcoral', 'gold']
  )
  plt.title('Overall Sentiment Distribution')
  plt.xlabel('Sentiment')
  plt.ylabel('Number of Reviews')
  plt.show()
  
  
  
def Calculate_sentiment_trends_over_time():
  df = get_sentiment_trends_data()
  sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
  df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)

  # Convert timestamp to datetime and extract monthly periods
  df['date'] = pd.to_datetime(df['timestamp'], unit='s')
  df['month'] = df['date'].dt.to_period('M')  # Group by month
  
  # Group by month and calculate average sentiment score
  monthly_sentiment = df.groupby('month')['sentiment_score'].mean().reset_index()

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
  
  # Create a pivot table for the heatmap (example assumes year and month columns)
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