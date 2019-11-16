select cast(min(p.pay) as float) / max(p.pay)
from(
    select count(distinct user_pseudo_id) as pay
    from(
            select user_pseudo_id from test_table_001 where event_name='unlock_content' and event_params_value_string_value='INAPP'
            union all
            select user_pseudo_id from test_table_002 where event_name='unlock_content' and event_params_value_string_value='INAPP'
            union all
            select user_pseudo_id from test_table_003 where event_name='unlock_content' and event_params_value_string_value='INAPP'
            union all
            select user_pseudo_id from test_table_004 where event_name='unlock_content' and event_params_value_string_value='INAPP'
            union all
            select user_pseudo_id from test_table_005 where event_name='unlock_content' and event_params_value_string_value='INAPP'
            union all
            select user_pseudo_id from test_table_006 where event_name='unlock_content' and event_params_value_string_value='INAPP'
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