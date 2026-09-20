import json
import os

# No key here - the client and models come from groq_client.py, which reads
# the single .env file in this folder.
from groq_client import client, MODEL as MODEL_NAME
# Load restaurant data
with open("California-Culinary-Map.txt", "r", encoding="utf-8") as file:
    data = file.read()

# Split the data into individual restaurant descriptions
restaurant_list = data.split("\n\n")

# Remove the dataset title
restaurant_list = restaurant_list[1:]

print("Number of restaurants:", len(restaurant_list))
# Example restaurant used to demonstrate the desired output format
example_restaurant = restaurant_list[1]

# Example of the structured JSON we want the model to produce
example_output = {
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "sun-drenched, salt-air energy",
    "environment": "open-air dining near the pier",
    "shortcomings": []
}

print("\nExample restaurant:")
print(example_restaurant)

print("\nExample structured output:")
print(json.dumps(example_output, indent=4))
# ==========================================
# Step 5: Create the restaurant prompt
# ==========================================

def create_restaurant_prompt(restaurant):

    prompt = f"""
You are an expert data extraction assistant.

Extract the restaurant information from the text below
and return ONLY a valid JSON object.

The JSON must contain these fields:

- name
- location
- type
- food_style
- rating
- price_range
- signatures
- vibe
- environment
- shortcomings

Rules:
- rating must be a number.
- Convert price range to a number:
  $ = 1
  $$ = 2
  $$$ = 3
  $$$$ = 4
- signatures must be a list of strings.
- shortcomings must be a list of strings.
- If information is not available, use null or an empty list.
- Do not add any fields that are not listed above.
- Do not include Markdown or explanations outside the JSON.

Here is an example:

Input:
{example_restaurant}

Output:
{json.dumps(example_output, indent=4)}

Now structure this restaurant:

Input:
{restaurant}
"""

    return prompt


# Test the prompt
test_prompt = create_restaurant_prompt(restaurant_list[0])

print("\n--- Generated Prompt ---")
print(test_prompt)
# Send the prompt to Groq
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": test_prompt
        }
    ]
)

print(response.choices[0].message.content)
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


def validate_restaurant_response(response_text):
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")

    clean_json = response_text[start:end + 1]
    clean_json = clean_json.replace("{{", "{").replace("}}", "}")

    restaurant = Restaurant.model_validate_json(clean_json)

    return restaurant


# Your normal validation test
try:
    restaurant = validate_restaurant_response(response.choices[0].message.content)

    print("\n--- Validation Successful ---")
    print(restaurant)

except (ValidationError, ValueError) as e:
    print("\n--- Validation Failed ---")
    print(e)


# Deliberately invalid test
invalid_json = """
{
    "name": "Test Restaurant",
    "location": "Los Angeles",
    "type": "bistro",
    "food_style": "Californian",
    "rating": "four point five",
    "price_range": 4,
    "signatures": [],
    "vibe": "modern",
    "environment": "casual",
    "shortcomings": []
}
"""

try:
    validate_restaurant_response(invalid_json)
    print("Unexpected: validation passed")

except ValidationError as e:
    print("\n--- Expected Validation Error ---")
    print(e)

def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
    You are a JSON repair expert. Your only job is to take a broken JSON string and a validation error message,
    and return a perfectly formatted, valid JSON object. Do not include any conversational text.
    """

    auto_repair_prompt = f"""
    The following JSON failed validation:
    {candidate_json_output}

    Error received: {error_message}

    Please provide the corrected JSON object:
    """

    return auto_repair_system_msg, auto_repair_prompt
structured_data = []

for i, restaurant in enumerate(restaurant_list):

    print(f"Processing restaurant {i + 1}/{len(restaurant_list)}")

    prompt = create_restaurant_prompt(restaurant)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    candidate_json = response.choices[0].message.content

    for attempt in range(3):

        try:
            restaurant_data = validate_restaurant_response(candidate_json)
            structured_data.append(restaurant_data)
            break

        except ValidationError as e:

            system_msg, repair_prompt = JSON_auto_repair_prompts(
                candidate_json,
                str(e)
            )

            repair_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": repair_prompt
                    }
                ]
            )

            candidate_json = repair_response.choices[0].message.content