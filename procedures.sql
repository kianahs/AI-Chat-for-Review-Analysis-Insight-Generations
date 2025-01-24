CREATE PROCEDURE GetSentimentDistribution()
BEGIN
SELECT COUNT(id) AS value, sentiment  
FROM `customers-reviews-database.amazon_db.Sentiments` 

GROUP BY sentiment;

END;


CREATE PROCEDURE GetSentimentTrendsData()
BEGIN

SELECT r.id, r.timestamp, s.sentiment
FROM 
        `customers-reviews-database.amazon_db.Reviews` r
    JOIN 
        `customers-reviews-database.amazon_db.Sentiments` s
    ON 
        r.id = s.id;
END;
