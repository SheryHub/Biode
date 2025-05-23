# gemini_bird_identifier.py

from google import genai
import os

class GeminiBirdIdentifier:
    def __init__(self, api_key: str, model: str = "models/gemini-1.5-flash"):
        """
        Initializes the Gemini client with the provided API key and model.
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def identify_bird(self, image_path: str) -> str:
        """
        Uploads an image and uses the Gemini model to identify the bird,
        returning the response text.
        """
        uploaded_file = self.client.files.upload(file=image_path)
        prompt = """
        Identify the bird in the image and provide the following details in JSON format:
        {
          "scientific_name": "",
          "common_name": "",
          "local_name": "",
          "classification_score": 0.5
        }
        Respond ONLY in JSON with no markdown . Keep the response short and concise and also no markdown just only json.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded_file, prompt]
        )

        return response.text


if __name__ == "__main__":

    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    bird_identifier = GeminiBirdIdentifier(api_key=API_KEY)
    image_path = "images/images.jpg"
    result = bird_identifier.identify_bird(image_path)

    # Print the result
    print(result)
