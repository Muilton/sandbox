SELECT 	STR_TO_DATE(revenue.date, '%m/%d/%Y') as period,
		app.name as app_name,
        AVG(revenue) as mean_revenue
FROM revenue, app
WHERE (STR_TO_DATE(revenue.date, '%m/%d/%Y') > '2018-01-14' AND
	   STR_TO_DATE(revenue.date, '%m/%d/%Y') < '2018-02-16' AND
	   revenue.app_id = app.app_id )
GROUP BY app.name