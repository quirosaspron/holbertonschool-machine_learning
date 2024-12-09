#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the number of launches per rocket as:
<rocket name>: <number of launches>
ordered by the number of launches in descending order or,
if rockets have the same amount of launches, in alphabetical order
"""

import requests
from collections import defaultdict

if __name__ == "__main__":
    # Fetch all launches
    launches_url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(launches_url).json()

    # Fetch all rockets
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    rockets = requests.get(rockets_url).json()

    # Create a mapping of rocket ID to rocket name
    rocket_id_to_name = {rocket['id']: rocket['name'] for rocket in rockets}

    # Count launches for each rocket
    rocket_launch_count = defaultdict(int)
    for launch in launches:
        rocket_id = launch.get('rocket')
        if rocket_id and rocket_id in rocket_id_to_name:
            rocket_name = rocket_id_to_name[rocket_id]
            rocket_launch_count[rocket_name] += 1

    # Sort rockets by launch count and name
    sorted_rockets = sorted(
        rocket_launch_count.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Print results
    for rocket_name, count in sorted_rockets:
        print(f"{rocket_name}: {count}")
