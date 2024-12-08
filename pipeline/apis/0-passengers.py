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
    ships = requests.get('https://swapi-api.hbtn.io/api/starships').json()
    ships = ships.get("results", [])
    ship_passengers = [[entry['name'], entry['passengers'].replace(',', '')]
                       for entry in ships]
    for ship in ship_passengers:
        if ship[1].isdigit() and int(ship[1]) >= passengerCount:
            qualified_ships.append(ship[0])
    return qualified_ships
