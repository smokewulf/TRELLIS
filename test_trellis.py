import requests

# URL of the FastAPI endpoint
url = "http://localhost:8000/generate_3d_model"

# Path to the image you want to send
image_path = "path/to/your/example_image.png"  # Update this to your image path

# Prepare the files parameter for the POST request
files = {"image_path": open(image_path, "rb")}

try:
    # Send POST request to the FastAPI endpoint
    response = requests.post(url, files=files)

    # Check if the request was successful
    if response.status_code == 200:
        print("Success!")
        print("Response JSON:", response.json())
    else:
        print("Error:", response.status_code)
        print("Response:", response.text)

finally:
    # Close the file handle
    files["image_path"].close()
