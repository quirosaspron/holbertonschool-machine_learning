#!/usr/bin/env python3
"""
returns the list of names of the home planets of all sentient species
"""
import requests


def sentientPlanets():
    """
    returns: list of names of the home planets of all sentient species
    """
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species'
    while url:
        # Fetch current page of results
        response = requests.get(url).json()

        # Extract species from current page
        species = response.get("results", [])
        sen = 'sentient'

        # Process the species
        for specie in species:
            if specie['designation'] == sen or specie['classification'] == sen:
                if specie['homeworld'] is not None:
                    if specie['homeworld'] not in planets:
                        home = requests.get(specie['homeworld']).json()
                        planets.append(home['name'])

        # Move to the next page
        url = response.get('next')

    return planets
