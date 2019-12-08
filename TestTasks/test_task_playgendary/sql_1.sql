select event_name, count(event_name)
from(
        select event_name, min(event_timestamp), user_pseudo_id
        from(
            select event_name, event_timestamp, user_pseudo_id from test_table_001 where event_name in ('gdpr_group_1', 'gdpr_group_2', 'gdpr_group_3', 'gdpr_group_4', 'gdpr_group_5', 'gdpr_group_6')
            union all
            select event_name, (event_timestamp), user_pseudo_id from test_table_002 where event_name in ('gdpr_group_1', 'gdpr_group_2', 'gdpr_group_3', 'gdpr_group_4', 'gdpr_group_5', 'gdpr_group_6') 
            union all
            select event_name, (event_timestamp), user_pseudo_id from test_table_003 where event_name in ('gdpr_group_1', 'gdpr_group_2', 'gdpr_group_3', 'gdpr_group_4', 'gdpr_group_5', 'gdpr_group_6') 
            union all
            select event_name, (event_timestamp), user_pseudo_id from test_table_004 where event_name in ('gdpr_group_1', 'gdpr_group_2', 'gdpr_group_3', 'gdpr_group_4', 'gdpr_group_5', 'gdpr_group_6') 
            union all
            select event_name, (event_timestamp), user_pseudo_id from test_table_005 where event_name in ('gdpr_group_1', 'gdpr_group_2', 'gdpr_group_3', 'gdpr_group_4', 'gdpr_group_5', 'gdpr_group_6')
            union all
            select event_name, (event_timestamp), user_pseudo_id from test_table_006 where event_name in ('gdpr_group_1', 'gdpr_group_2', 'gdpr_group_3', 'gdpr_group_4', 'gdpr_group_5', 'gdpr_group_6')
        )
        group by user_pseudo_id
    )
group by event_name    
