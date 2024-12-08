#!/usr/bin/env python3
"""
By using the GitHub API, we print the location of a specific user
"""
if __name__ == '__main__':
    import requests
    import sys
    from datetime import datetime, timedelta

    url = sys.argv[1]
    request = requests.get(url)
    if request.status_code == 404:
        print('Not found')

    if request.status_code == 403:
        minutes_to_reset = int(request.headers.get("X-RateLimit-Reset"))
        now = datetime.now()
        reset_time = now + timedelta(minutes=minutes_to_reset)
        print(f'Reset in {reset_time} min')

    if request.status_code == 202:
        response = request.json()
        location = response['location']
        print(location)
