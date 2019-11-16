SELECT campaign.name, sum(revenue) 
FROM revenue, app, campaign 
WHERE (STR_TO_DATE(revenue.date, '%m/%d/%Y') > '2018-01-31' AND 
		revenue.app_id = app.app_id AND 
		app.campaign_id = campaign.id) 
GROUP BY campaign.name 
ORDER BY sum(revenue) DESC LIMIT 1