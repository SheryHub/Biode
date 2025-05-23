import json
import re
import os
from classify import GeminiBirdIdentifier

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    print("Please provide a an api key via API_KEY environment variable")
    return
bird_identifier = GeminiBirdIdentifier(api_key=API_KEY)

image_path = "images/images.jpg"  # replace with your local image path

result_json = bird_identifier.identify_bird(image_path)
print("Raw response:")
print(result_json)

# Clean the result using regex
match = re.search(r"\{.*\}", result_json, re.DOTALL)
if match:
    cleaned_json_str = match.group(0)
    try:
        result = json.loads(cleaned_json_str)
        print("Parsed JSON:")
        print(result)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
else:
    print("No JSON object found in the response.")
