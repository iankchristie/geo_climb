import requests

url = "https://macrostrat.org/api/v2/mobile/map_query_v2"

# z is zoom, testing on the map shows that 12 seems sufficient.
params = {"lng": -105.2705, "lat": 40.0150, "z": 12}

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
