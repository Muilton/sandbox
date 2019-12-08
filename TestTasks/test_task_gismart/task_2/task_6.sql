SELECT 	WEEK(STR_TO_DATE(revenue.date, '%m/%d/%Y'), 1) as week,
        SUM(revenue) as revenue_by_week
FROM revenue
GROUP BY WEEK(STR_TO_DATE(revenue.date, '%m/%d/%Y'), 1)
ORDER BY revenue_by_week DESC LIMIT 1