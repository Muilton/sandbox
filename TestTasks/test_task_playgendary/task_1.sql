select * 
from (
        select * from test_table_001
        union all
        select * from test_table_002 
        union all
        select * from test_table_003 
        union all
        select * from test_table_004
        union all
        select * from test_table_005
        union all
        select * from test_table_006
    )as p0
where p0.event_name='match_start' and p0.event_params_key='content_type'