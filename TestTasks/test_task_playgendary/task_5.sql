select pay as all_unique_users, cast(min(p.pay) as float) / max(p.pay) as part_of_users_watch
from(
    select count(distinct user_pseudo_id) as pay
    from(
            select user_pseudo_id from test_table_001 where event_name='video_show'
            union all
            select user_pseudo_id from test_table_002 where event_name='video_show'
            union all
            select user_pseudo_id from test_table_003 where event_name='video_show'
            union all
            select user_pseudo_id from test_table_004 where event_name='video_show'
            union all
            select user_pseudo_id from test_table_005 where event_name='video_show'
            union all
            select user_pseudo_id from test_table_006 where event_name='video_show'
        )
    
    union all
    
    select count(distinct user_pseudo_id)
    from(
            select user_pseudo_id from test_table_001
            union all
            select user_pseudo_id from test_table_002 
            union all
            select user_pseudo_id from test_table_003 
            union all
            select user_pseudo_id from test_table_004 
            union all
            select user_pseudo_id from test_table_005 
            union all
            select user_pseudo_id from test_table_006 
    )
) as p