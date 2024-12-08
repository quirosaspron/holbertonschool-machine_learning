#!/usr/bin/env python3
"""
returns the list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    passangerCount: Minimum number of passangers
    returns: the ships that are qualified based on passanger count
    """
    qualified_ships = []
    url = 'https://swapi-api.hbtn.io/api/starships'
    while url:
        # Fetch current page of results
        response = requests.get(url).json()

        # Extract ships from current page
        ships = response.get("results", [])

        # Process the ships
        for ship in ships:
            passengers = ship['passengers'].replace(',', '')
            if passengers.isdigit():
                if int(passengers) >= passengerCount:
                    qualified_ships.append(ship['name'])

        # Move to the next page
        url = response.get('next')

    return qualified_ships
