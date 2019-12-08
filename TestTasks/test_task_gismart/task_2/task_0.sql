SELECT date,
sum(revenue) 

FROM gismart_test.revenue

group by date