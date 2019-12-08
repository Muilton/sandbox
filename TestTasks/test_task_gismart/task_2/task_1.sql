SELECT date, sum(revenue) 
FROM revenue 
GROUP BY date
ORDER BY sum(revenue) DESC LIMIT 1