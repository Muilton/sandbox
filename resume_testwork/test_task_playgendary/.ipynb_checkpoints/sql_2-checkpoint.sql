select event_date, count(DISTINCT user_pseudo_id), count(user_pseudo_id), count(user_pseudo_id) / count(DISTINCT user_pseudo_id) as mean
from (
    select p0.user_pseudo_id, p0.event_date, p0.event_timestamp, p0.event_name, p1.event_name, p1.event_timestamp, (cast(p1.event_timestamp as int) - cast(p0.event_timestamp as int))/1000000 as delta 
    from (
        select *
        from (
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_001
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_002
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_003
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_004 
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_005
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_006
                )
        ) as p0
    join (
        select *
        from (
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_001 
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_002 
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_003 
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_004 
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_005
                union all
                select event_date, event_timestamp, event_name, user_pseudo_id from test_table_006
            )
        ) as p1 on p0.user_pseudo_id=p1.user_pseudo_id
    where p0.event_name='first_open' and p1.event_name='video_show' and (cast(p1.event_timestamp as int) - cast(p0.event_timestamp as int))/1000000 < 3600 and (cast(p1.event_timestamp as int) - cast(p0.event_timestamp as int))/1000000 > 0 
)
group by event_date