### 1.1: Define the file_path to the text file
file_path = "California-Culinary-Map.txt"


### 1.2: Open the text file
with open(file_path, "r") as file:
    data = file.read()


### 1.3: Print the first 100 characters of the restaurant data
print(data[:100])


### 2.1: Split the restaurant paragraphs into a list
restaurant_list = data.split("\n\n")


### 2.2: Since the first item is the dataset name, remove it
restaurant_list = restaurant_list[1:]


### 2.3: Print the number of restaurants
print(len(restaurant_list))


### 2.4: Print the first restaurant
print(restaurant_list[0])
# ==========================================
# Exercise 2: Connect to Groq
# ==========================================

# No key here - the client and model come from groq_client.py, which reads the
# single .env file in this folder.
from groq_client import client, MODEL as MODEL_NAME


# Function to send a prompt to Groq
def generate_response(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# Test the connection
test_prompt = "Give me one sentence describing a restaurant."

response = generate_response(test_prompt)

print("\nGroq response:")
print(response)