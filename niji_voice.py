

import requests

url = "https://api.nijivoice.com/api/platform/v1/voice-actors"
key = ''

headers = {
    "accept": "application/json",
    "x-api-key": key
}

response = requests.get(url, headers=headers)

print(response.text)
