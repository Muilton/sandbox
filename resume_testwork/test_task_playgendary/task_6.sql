select *
    from(
            select * from test_table_001 where event_name in ('unlock_content', 'match_start', 'video_show', 'Sessionstart_with_internet', 'spend_ingame_currency')
            union all
            select * from test_table_002 where event_name in ('unlock_content', 'match_start', 'video_show', 'Sessionstart_with_internet', 'spend_ingame_currency')
            union all
            select * from test_table_003 where event_name in ('unlock_content', 'match_start', 'video_show', 'Sessionstart_with_internet', 'spend_ingame_currency')
            union all
            select * from test_table_004 where event_name in ('unlock_content', 'match_start', 'video_show', 'Sessionstart_with_internet', 'spend_ingame_currency')
            union all
            select * from test_table_005 where event_name in ('unlock_content', 'match_start', 'video_show', 'Sessionstart_with_internet', 'spend_ingame_currency')
            union all
            select * from test_table_006 where event_name in ('unlock_content', 'match_start', 'video_show', 'Sessionstart_with_internet', 'spend_ingame_currency')
        )