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

CREATE PROCEDURE `customers-reviews-database.amazon_db.GetOverallSentimentDistributionPerProductID` (IN i_id STRING)
BEGIN
    SELECT 
        COUNT(S.id) AS value, 
        S.sentiment  
    FROM `customers-reviews-database.amazon_db.Sentiments` AS S
    JOIN `customers-reviews-database.amazon_db.Reviews` AS R
    ON S.id = R.id
    WHERE R.parent_asin = i_id
    GROUP BY S.sentiment;
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