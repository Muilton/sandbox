SELECT 	MONTH(STR_TO_DATE(revenue.date, '%m/%d/%Y')) as month,
		WEEK(STR_TO_DATE(revenue.date, '%m/%d/%Y'), 1) as week,
		campaign.name as company_name,
        sum(revenue)
FROM revenue, app, campaign
WHERE (revenue.app_id = app.app_id AND
       app.campaign_id = campaign.id)
GROUP BY MONTH(STR_TO_DATE(revenue.date, '%m/%d/%Y')),
		 WEEK(STR_TO_DATE(revenue.date, '%m/%d/%Y'), 1), 
         campaign.name