import cv2
import base64
from openai import OpenAI
import re
import json

def analyze_image_with_gpt4o(frame, prompt, api_key=None):
    if api_key is None:
        with open("api_key.txt", "r") as file:
            api_key = file.read().strip()

    # Encode frame as JPEG
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise ValueError("Failed to encode the image.")

    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Create the request with Vision
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=1000  # Optional: control response length
    )

    # Return the raw text response
    text_response = response.choices[0].message.content
    # Write the response to a file for debugging
    with open("response.md", "w") as response_file:
        response_file.write(text_response)
    return text_response

def extract_json_objects(text):
    # Basic regex pattern for a JSON object
    # This pattern is quite naive and might not work for all valid JSON objects
    pattern = r'\{[^\{]*?\}'
    
    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Try to parse each match as JSON
    json_objects = []
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Skip if it's not a valid JSON object
            continue
    
    return json_objects

if __name__ == "__main__":
    # Example usage
    frame = cv2.imread(r"resources\image.jpg")  # Replace with your image
    prompt = "What animal is in this image? Return a JSON like: {\"animal\": \"dog\"}"
    try:
        text_response = analyze_image_with_gpt4o(frame, prompt)

        json_objects = extract_json_objects(text_response)
        if json_objects:
            print(json_objects[0])
        else:
            print("No valid JSON objects found in the response.")
            print(text_response)

            # Save the response to a file
            with open("response.txt", "w") as response_file:
                response_file.write(text_response)
    except Exception as e:
        print(f"Error: {e}")