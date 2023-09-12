import requests

url = "https://api-nba-v1.p.rapidapi.com/games"

querystring = {"id":"8899"}

headers = {
	"X-RapidAPI-Key": "a16e5cb61emsh7dc4ccb88902efep1ede33jsn51bcf79dd912",
	"X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())