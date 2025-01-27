CREATE PROCEDURE `customers-reviews-database.amazon_db.GetOverallSentimentDistribution`()
BEGIN
SELECT COUNT(id) AS value, sentiment  
FROM `customers-reviews-database.amazon_db.Sentiments` 

GROUP BY sentiment;

END;


CREATE PROCEDURE `customers-reviews-database.amazon_db.GetOverallSentimentTrendsData`()
BEGIN

SELECT r.id, r.timestamp, s.sentiment
FROM 
        `customers-reviews-database.amazon_db.Reviews` r
    JOIN 
        `customers-reviews-database.amazon_db.Sentiments` s
    ON 
        r.id = s.id;
END;



CREATE PROCEDURE GetReviewsBySentiment(IN VARCHAR(256) i_sentiment)
BEGIN

SELECT r.id, r.text
FROM 
        `customers-reviews-database.amazon_db.Reviews` r
    JOIN 
        `customers-reviews-database.amazon_db.Sentiments` s
    ON 
        r.id = s.id
WHERE s.sentiment = i_sentiment;

END;