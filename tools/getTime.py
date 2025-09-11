import datetime

INTERVALS_SHORT = [
    (31536000, 'yrs'),
    (2592000, 'mos'),
    (604800, 'wks'),
    (86400, 'days'),
    (3600, 'hrs'),
    (60, 'mins'),
    (1, 'secs')
]

INTERVALS = [
    (31536000, 'years'),
    (2592000, 'months'),
    (604800, 'weeks'),
    (86400, 'days'),
    (3600, 'hours'),
    (60, 'minutes'),
    (1, 'seconds')
]

def humanTimeDiff(past_time, current_time=None):
    """
    Calculate the human-readable time difference between two datetime objects.
    
    Args:
        past_time: The past time (datetime object)
        current_time: The current time (datetime object), defaults to now

    Returns:
        str: A human-readable time difference string
    """
    if current_time is None:
        current_time = datetime.datetime.now()
    
    if past_time > current_time:
        past_time, current_time = current_time, past_time
    
    diff = current_time - past_time
    seconds = int(diff.total_seconds())
    


    

    if seconds < 10:
        return 'a few seconds ago'
    elif seconds < 60:
        return f'{seconds} seconds ago'

    for unit_seconds, unit_name in INTERVALS:
        count = seconds // unit_seconds
        if count > 0:
            if count == 1:
                return f'a {unit_name} ago'
            else:
                return f'{count} {unit_name} ago'
    
    return 'just now'


if __name__ == "__main__":
    now = datetime.datetime.now()
    

    tests = [
        ("5 sec", now - datetime.timedelta(seconds=5)),
        ("30 sec", now - datetime.timedelta(seconds=30)),
        ("2 min", now - datetime.timedelta(minutes=2)),
        ("1 hr", now - datetime.timedelta(hours=1)),
        ("3 hr", now - datetime.timedelta(hours=3)),
        ("2 days", now - datetime.timedelta(days=2)),
        ("5 days", now - datetime.timedelta(days=5)),
        ("2 weeks", now - datetime.timedelta(weeks=2)),
        ("1 month", now - datetime.timedelta(days=30)),
        ("3 months", now - datetime.timedelta(days=90)),
        ("1 year", now - datetime.timedelta(days=365)),
    ]
    
    for desc, test_time in tests:
        result = humanTimeDiff(test_time)
        print(f"{desc}: {result}")