import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# ==========================================
# Exercise 1: Preprocess the food recipe data
# Step 1: Explore the food recipe JSON file
#       and its images
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RECIPE_FILE = BASE_DIR / "Recipes.json"
IMAGE_DIR = BASE_DIR / "synthetic_recipe_images"


# Step 1.1: Load the JSON file
with open(RECIPE_FILE, "r", encoding="utf-8") as file:
    recipe_data = json.load(file)


# First recipe
recipe1 = recipe_data[0]


# Step 1.2: Print each key-value pair of the first recipe
for key, value in recipe1.items():
    print(f"{key} ({type(value).__name__}): {value}")


# Step 1.3: Show the image of the first recipe
recipe_id = recipe1["id"]
image_path = IMAGE_DIR / f"recipe{recipe_id}.png"

print(f"\nImage path: {image_path}")

image = Image.open(image_path)

plt.imshow(image)
plt.axis("off")
plt.show()