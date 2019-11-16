select event_date, count(DISTINCT user_pseudo_id), count(user_pseudo_id), cast(count(user_pseudo_id) as float) / count(DISTINCT user_pseudo_id) as mean
from(
            select event_date, event_name, user_pseudo_id from test_table_001 where event_name in ('Sessionstart_with_internet', 'Sessionstart_with_nointernet')
            union all
            select event_date, event_name, user_pseudo_id from test_table_002 where event_name in ('Sessionstart_with_internet', 'Sessionstart_with_nointernet')
            union all
            select event_date, event_name, user_pseudo_id from test_table_003 where event_name in ('Sessionstart_with_internet', 'Sessionstart_with_nointernet')
            union all
            select event_date, event_name, user_pseudo_id from test_table_004 where event_name in ('Sessionstart_with_internet', 'Sessionstart_with_nointernet')
            union all
            select event_date, event_name, user_pseudo_id from test_table_005 where event_name in ('Sessionstart_with_internet', 'Sessionstart_with_nointernet')
            union all
            select event_date, event_name, user_pseudo_id from test_table_006 where event_name in ('Sessionstart_with_internet', 'Sessionstart_with_nointernet')
    )
group by event_date   
