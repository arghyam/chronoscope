import base64
import os
import logging
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv

# Add these lines at the start after imports
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

load_dotenv()

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_image(image_path, api_key):
    """Process an image using Gemini 2.0 Flash."""
    # Get base64 encoded image
    image_data = encode_image(image_path)
    if not image_data:
        return None

    # Configure Google AI
    configure(api_key=api_key)

    # Initialize Gemini model
    model = GenerativeModel('gemini-2.0-flash')

    # Process with Gemini
    try:
        response = model.generate_content(
            contents=[
                "Please extract only the numbers from this image",
                {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
            ],
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 1024,
            }
        )
        
        return response.text if response.text else "No numbers found."
    
    except Exception as e:
        return f"Error processing image: {e}"

def main():
    # Path to your image
    image_path = "output/straightened.jpg"
    
    # Get API key from environment variables
    api_key = os.environ["GOOGLE_API_KEY"]
    
    # Process the image
    result = process_image(image_path, api_key)
    
    # Print results
    print("OCR Result:")
    print(result)
    
    # Optionally save results to file
    with open("ocr_result.txt", "w", encoding="utf-8") as f:
        f.write(result)

if __name__ == "__main__":
    main()