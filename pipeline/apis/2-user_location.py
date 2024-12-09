#!/usr/bin/env python3
"""
print the location of a specific user or rate limit reset
"""
if __name__ == '__main__':
    import requests
    import sys
    from datetime import datetime

    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        # Make the request
        request = requests.get(url)

        # Handle 404: Not found
        if request.status_code == 404:
            print('Not found')
            sys.exit(1)

        # Handle 403: Rate limit exceeded
        if request.status_code == 403:
            reset_time_unix = int(request.headers.get("X-RateLimit-Reset", 0))
            now_unix = int(datetime.now().timestamp())
            minutes_to_reset = (reset_time_unix - now_unix) // 60
            print(f'Reset in {minutes_to_reset} min')
            sys.exit(1)

        # Handle successful response
        if request.status_code == 200:
            response = request.json()
            location = response.get('location', 'Location not available')
            print(location)

    except requests.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)
