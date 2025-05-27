import threading
import cv2
import base64
from openai import OpenAI
import re
import json
import os
import time

def analyze_image_with_gpt4o(frame, prompt, api_key=None):
    # Read API key from api_key.txt in the script's parent folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    api_key_path = os.path.join(parent_dir, "api_key.txt")
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()

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

    # Prepare interaction log and image saving
    timestamp = int(time.time() * 1000)
    interaction_dir = "./interactions"
    os.makedirs(interaction_dir, exist_ok=True)
    image_filename = f"{timestamp}.png"
    text_filename = f"{timestamp}.md"
    image_path = os.path.join(interaction_dir, image_filename)
    text_path = os.path.join(interaction_dir, text_filename)

    # Save the passed frame as PNG
    cv2.imwrite(image_path, frame)

    # Build the interaction string including the image path (using markdown image syntax)
    interaction = f"# Input\n{prompt}\n\n" \
              f"![Input Image]({image_filename})\n\n" \
              f"# Output\n{text_response}"

    # Save the interaction log to a markdown file
    with open(text_path, "w") as response_file:
        response_file.write(interaction)

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
    pass
