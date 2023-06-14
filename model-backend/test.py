import requests

# Test rice model
url_rice = 'https://get-prediction-rfi66hequa-et.a.run.app/index/rice'
image_path_rice = 'padi-sehat.jpg'
files_rice = {'file': open(image_path_rice, 'rb')}
response_rice = requests.post(url_rice, files=files_rice)
print(response_rice.json())

# Test corn model
url_corn = 'https://get-prediction-rfi66hequa-et.a.run.app/index/maize'
image_path_corn = 'jagung-sakit.jpg'
files_corn = {'file': open(image_path_corn, 'rb')}
response_corn = requests.post(url_corn, files=files_corn)
print(response_corn.json())