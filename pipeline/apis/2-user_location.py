#!/usr/bin/env python3
"""
By using the GitHub API, we print the location of a specific user
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
            reset_time = datetime.fromtimestamp(reset_time_unix)
            print(f'Reset in {reset_time.strftime("%Y-%m-%d %H:%M:%S")} UTC')
            sys.exit(1)

        # Handle successful response
        if request.status_code == 200:
            response = request.json()
            location = response.get('location', 'Location not available')
            print(location)

    except requests.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)

