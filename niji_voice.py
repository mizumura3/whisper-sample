

import requests

url = "https://api.nijivoice.com/api/platform/v1/voice-actors"
key = 'ad6b7f8b-4ee5-4b50-99cf-8ea8d31b2cb2'

headers = {
    "accept": "application/json",
    "x-api-key": key
}

response = requests.get(url, headers=headers)

print(response.text)
