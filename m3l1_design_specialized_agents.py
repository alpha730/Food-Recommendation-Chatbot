"""
M3L1 - Design Specialized Agents

Auto-extracted verbatim from: M3L1_Design_Specialized_Agents_2.ipynb
Code cells are unchanged; notebook magics are commented out.
"""


# ============================================================================
# Design Specialized Agents for a Recommendation System | Learning Objectives | Introduction
# ============================================================================

# %%capture
# %pip install openai==1.99.9 | tail -n 1
# %pip install langchain==0.3.0 | tail -n 1
# %pip install langchain-community==0.3.27 | tail -n 1 
# %pip install langchain-core==0.3.74 | tail -n 1 
# %pip install langchain_experimental==0.3.4 | tail -n 1 
# %pip install openai==1.99.9 | tail -n 1


# ============================================================================
# Import required libraries
# ============================================================================

import os                         
from typing import List, Dict, Any                           
# Swapped from OpenAI to Groq - see groq_client.py
from groq_client import client, MODEL, HumanMessage, SystemMessage


# ============================================================================
# Configure API access
# ============================================================================

# Set your Groq API key in the environment before running:
#   set GROQ_API_KEY=...      (Windows)  /  export GROQ_API_KEY=...
# `client` and `MODEL` come from groq_client.py, imported above.


# ============================================================================
# Define agent roles, goals, and backstories | Agent 1: User Profile Generator
# ============================================================================

user_profile_agent_config = {
    "role": "User Profile Generator",
    "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile including preferences, dietary restrictions, favorite cuisines, and dining patterns.",
    "backstory": """You are an expert user behavior analyst with 10 years of experience in the food and hospitality industry. 
    You excel at reading between the lines to understand not just what users say they like, but what their actions reveal 
    about their true preferences. You have a talent for identifying patterns in dining behavior, recognizing subtle preferences, 
    and building rich user profiles that capture both explicit and implicit food preferences. You understand that a user's 
    social media posts and check-ins tell a story about their culinary journey, and you're skilled at extracting meaningful 
    insights from unstructured data."""
}

print(f"Role: {user_profile_agent_config['role']}")
print(f"\nGoal: {user_profile_agent_config['goal']}")
print(f"\nBackstory: {user_profile_agent_config['backstory']}")


# ============================================================================
# Agent 2: RAG Retriever
# ============================================================================

rag_retriever_agent_config = {
    "role": "RAG Retriever",
    "goal": "Query multimodal vector databases to retrieve relevant restaurants, recipes, and food-related content based on user profiles and similarity search.",
    "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search. 
    You understand how embeddings capture meaning and can craft queries that retrieve the most relevant information 
    from large collections of restaurant data, recipes, and food images. You know when to use similarity search versus 
    filtered search, and you can balance relevance with diversity to ensure recommendations aren't repetitive. 
    You've worked with Pinecone, Weaviate, and ChromaDB, and you understand the nuances of multimodal retrieval 
    where text and images work together to represent food experiences."""
}

print(f"Role: {rag_retriever_agent_config['role']}")
print(f"\nGoal: {rag_retriever_agent_config['goal']}")
print(f"\nBackstory: {rag_retriever_agent_config['backstory']}")


# ============================================================================
# Agent 3: Food Trend Analyst
# ============================================================================

food_trend_analyst_config = {
    "role": "Food Trend Analyst",
    "goal": "Identify current food trends, popular ingredients, emerging dining concepts, and culinary movements to ensure recommendations are timely and culturally relevant.",
    "backstory": """You are a culinary journalist and trend forecaster who has spent 15 years covering food culture across 
    global markets. You have your finger on the pulse of what's happening in the food world—from viral TikTok recipes to 
    Michelin-starred innovations. You track emerging ingredients like kelp noodles and yuzu, monitor the rise of food 
    movements like plant-based dining and zero-waste cooking, and spot the next big thing before it goes mainstream. 
    You read Eater, Bon Appétit, and industry reports daily, and you know the difference between a fleeting fad and a 
    lasting trend."""
}

print(f"Role: {food_trend_analyst_config['role']}")
print(f"\nGoal: {food_trend_analyst_config['goal']}")
print(f"\nBackstory: {food_trend_analyst_config['backstory']}")


# ============================================================================
# Agent 4: Food Style Expert | **Question 1**: Complete the Food Style Expert configuration below by filling in the `goal` field.
# ============================================================================

food_style_expert_config = {
    "role": "Food Style Expert",
    "goal": "Analyze cuisine types, regional variations, cooking methods, and flavor profiles of the retrieved restaurants and recipes, and match them to the user's taste preferences so that each recommendation is grounded in a clear culinary rationale.",  # Fill this in
    "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines. 
    You've cooked in kitchens across five continents and understand the techniques, ingredients, and cultural contexts 
    that define different food traditions. You can distinguish Sichuan from Cantonese, Neapolitan pizza from Roman, 
    and Nashville hot chicken from Buffalo wings. You understand flavor profiles—umami-rich, bright and acidic, 
    rich and creamy—and can map them to user preferences. You respect culinary heritage while staying open to fusion 
    and innovation."""
}

print(f"Role: {food_style_expert_config['role']}")
print(f"\nGoal: {food_style_expert_config['goal']}")
print(f"\nBackstory: {food_style_expert_config['backstory']}")


# ============================================================================
# Agent 5: Nutrition Expert
# ============================================================================

nutrition_expert_config = {
    "role": "Nutrition Expert",
    "goal": "Evaluate nutritional content, identify allergens, assess dietary restrictions, and ensure recommendations align with users' health and wellness goals.",
    "backstory": """You are a registered dietitian with a master's degree in nutrition science and 8 years of clinical experience. 
    You understand macronutrients, micronutrients, and how different diets (keto, Mediterranean, plant-based, etc.) affect health. 
    You can quickly assess whether a dish fits within dietary restrictions like gluten-free, dairy-free, or low-sodium. 
    You're also sensitive to food allergies and intolerances, and you know how to balance health considerations with the 
    pleasure of eating. You believe that good nutrition doesn't mean sacrificing flavor or enjoyment."""
}

print(f"Role: {nutrition_expert_config['role']}")
print(f"\nGoal: {nutrition_expert_config['goal']}")
print(f"\nBackstory: {nutrition_expert_config['backstory']}")


# ============================================================================
# Agent 6: Recommendation Expert
# ============================================================================

recommendation_expert_config = {
    "role": "Recommendation Expert",
    "goal": "Synthesize insights from all agents—user profiles, retrieved data, trends, food styles, and nutrition—into cohesive, well-reasoned restaurant and recipe recommendations.",
    "backstory": """You are a recommendation systems architect with experience building personalization engines for major 
    food delivery platforms and recipe apps. You understand how to balance multiple signals—relevance, diversity, novelty, 
    and serendipity—to create recommendations that delight users. You know when to play it safe with familiar favorites 
    and when to suggest something unexpected. You're skilled at synthesizing complex, sometimes conflicting information 
    from multiple sources into clear, actionable recommendations. You write in a warm, engaging tone that makes users 
    excited to try new restaurants and recipes."""
}

print(f"Role: {recommendation_expert_config['role']}")
print(f"\nGoal: {recommendation_expert_config['goal']}")
print(f"\nBackstory: {recommendation_expert_config['backstory']}")


# ============================================================================
# **Question 2**: Which agent is responsible for querying the vector database built in Module 2? | **Answer**: You can use this cell to enter your answer. | Create agent instances
# ============================================================================

def create_agent_prompt(agent_config: Dict[str, str]) -> str:
    """Create a system prompt for an agent based on its configuration."""
    prompt = f"""You are a {agent_config['role']}.
    
Your goal: {agent_config['goal']}

Your background: {agent_config['backstory']}

Always respond in a professional, helpful manner that reflects your expertise.
"""
    return prompt

# Create agent prompts
user_profile_prompt = create_agent_prompt(user_profile_agent_config)
rag_retriever_prompt = create_agent_prompt(rag_retriever_agent_config)
food_trend_prompt = create_agent_prompt(food_trend_analyst_config)
food_style_prompt = create_agent_prompt(food_style_expert_config)
nutrition_prompt = create_agent_prompt(nutrition_expert_config)
recommendation_prompt = create_agent_prompt(recommendation_expert_config)

print("Agent prompts created successfully!")

print("=" * 80)
print("FOOD TREND ANALYST PROMPT")
print("=" * 80)
print(food_trend_prompt)


# ============================================================================
# Design agent tasks | Task 1: Generate the user profile
# ============================================================================

task_generate_profile = {
    "description": """Analyze the user's restaurant visit history and social media posts to create a comprehensive profile.
    Extract the following information:
    - Favorite cuisines and cuisine categories
    - Dietary restrictions or preferences (vegetarian, vegan, gluten-free, etc.)
    - Preferred dining occasions (casual, fine dining, quick bites)
    - Price sensitivity
    - Adventurousness (comfort food lover vs. culinary explorer)
    - Flavor preferences (spicy, sweet, savory, etc.)
    - Frequency of dining out
    
    Provide specific examples from the user's history to support each insight.""",
    
    "expected_output": """A structured user profile in JSON format with keys: 
    favorite_cuisines, dietary_restrictions, dining_occasions, price_range, 
    adventurousness_score (1-10), flavor_preferences, dining_frequency.
    Include a summary paragraph explaining the user's dining personality.""",
    
    "agent": "User Profile Generator"
}

print(f"Task: {task_generate_profile['description'][:100]}...")
print(f"\nExpected Output: {task_generate_profile['expected_output'][:100]}...")
print(f"\nAgent: {task_generate_profile['agent']}")


# ============================================================================
# Task 2: Retrieve relevant restaurants and recipes
# ============================================================================

task_retrieve_candidates = {
    "description": """Based on the user profile, query the vector database to retrieve:
    - Top 20 restaurants that match the user's preferences
    - Top 20 recipes that align with their taste and dietary needs
    
    Use similarity search with the user's favorite cuisines and flavor preferences as the query.
    Apply filters for dietary restrictions, price range, and location (if provided).
    Ensure diversity in the results—don't retrieve 20 Italian restaurants if the user likes multiple cuisines.""",
    
    "expected_output": """Two lists in JSON format:
    - restaurants: Array of restaurant objects with fields: name, cuisine_type, price_range, rating, description
    - recipes: Array of recipe objects with fields: name, cuisine_type, difficulty, prep_time, ingredients, description""",
    
    "agent": "RAG Retriever"
}

print(f"Task: {task_retrieve_candidates['description'][:100]}...")
print(f"\nExpected Output: {task_retrieve_candidates['expected_output'][:100]}...")
print(f"\nAgent: {task_retrieve_candidates['agent']}")


# ============================================================================
# Task 3: Analyze food trends
# ============================================================================

task_analyze_trends = {
    "description": """Analyze current food trends relevant to the retrieved restaurants and recipes.
    Identify:
    - Trending ingredients or techniques in the retrieved items
    - Popular dining concepts or restaurant types
    - Emerging culinary movements that align with the user's interests
    - Seasonal trends or timely food moments
    
    Provide context on why these trends matter and how they enhance the recommendations.""",
    
    "expected_output": """A trends analysis with:
    - List of 3-5 relevant trends with descriptions
    - Explanation of how each trend relates to the user's profile
    - Suggestions for which restaurants or recipes align with these trends""",
    
    "agent": "Food Trend Analyst"
}


# ============================================================================
# **Task 4**: Design the task for the Food Style Expert
# ============================================================================

## Type your answer here

task_analyze_food_styles = {
    "description": """Analyze the cuisine types, culinary techniques, and flavor profiles of the retrieved
    restaurants and recipes, and assess how well each one matches the user's taste preferences.
    For the candidate items, identify:
    - The cuisine type and its regional variation (e.g. Sichuan vs. Cantonese, Neapolitan vs. Roman)
    - The dominant cooking methods and techniques (grilling, braising, fermenting, raw preparations, etc.)
    - The core flavor profile (spicy, umami-rich, bright and acidic, rich and creamy, smoky, sweet)
    - Signature ingredients and how they define the dish or restaurant
    - Whether the item is traditional, modern, or fusion in style

    Then map these culinary attributes back to the user's profile: state which styles align strongly with
    their stated flavor preferences and adventurousness score, which ones stretch them in an interesting
    direction, and which ones are likely a poor stylistic match. Support every judgement with a specific
    culinary reason rather than a generic statement.""",  # Fill this in

    "expected_output": """A food style analysis in structured form containing:
    - style_breakdown: For each retrieved restaurant and recipe, an object with cuisine_type,
      regional_variation, cooking_methods, flavor_profile, signature_ingredients, and
      traditional_or_fusion
    - match_assessment: A match score (1-10) per item with a 1-2 sentence culinary justification
      explaining how it fits the user's flavor preferences
    - strong_matches: The items whose style best aligns with the user's palate
    - stretch_options: Items that differ from the user's usual choices but are worth trying, with a
      reason why the user is likely to enjoy them
    - poor_matches: Items to deprioritize, with the stylistic reason
    - A short summary paragraph describing the user's overall culinary style profile""",  # Fill this in

    "agent": "Food Style Expert"
}

print(f"Task: {task_analyze_food_styles['description'][:100]}...")
print(f"\nExpected Output: {task_analyze_food_styles['expected_output'][:100]}...")
print(f"\nAgent: {task_analyze_food_styles['agent']}")


# ============================================================================
# Task 5: Evaluate nutrition and dietary fit
# ============================================================================

task_evaluate_nutrition = {
    "description": """Evaluate the nutritional aspects of the retrieved restaurants and recipes.
    Check for:
    - Alignment with dietary restrictions (vegetarian, vegan, gluten-free, etc.)
    - Potential allergens or ingredients to avoid
    - Nutritional balance (protein, vegetables, whole grains)
    - Healthfulness relative to the user's goals
    
    Flag any items that don't meet the user's dietary needs and explain why.""",
    
    "expected_output": """A nutrition evaluation with:
    - List of items that meet all dietary restrictions
    - Items flagged for potential concerns (allergens, restrictions)
    - Nutritional highlights (high protein, vegetable-rich, etc.)
    - Overall assessment of how well the options support the user's health goals""",
    
    "agent": "Nutrition Expert"
}


# ============================================================================
# Task 6: Generate final recommendations
# ============================================================================

task_generate_recommendations = {
    "description": """Synthesize insights from all previous agents to generate final recommendations.
    Create:
    - Top 5 restaurant recommendations with detailed explanations
    - Top 5 recipe recommendations with detailed explanations
    
    For each recommendation, explain:
    - Why it matches the user's profile
    - How it aligns with current trends (if applicable)
    - What makes it a great fit in terms of food style
    - Any nutritional benefits or considerations
    
    Write in an engaging, enthusiastic tone that makes the user excited to try these options.""",
    
    "expected_output": """A recommendations report with:
    - restaurants: Array of 5 restaurant recommendations with name, description, and detailed reasoning
    - recipes: Array of 5 recipe recommendations with name, description, and detailed reasoning
    - Each recommendation should include a personalized explanation (2-3 sentences) of why it's a great match""",
    
    "agent": "Recommendation Expert"
}

print(f"Task: {task_generate_recommendations['description'][:100]}...")
print(f"\nExpected Output: {task_generate_recommendations['expected_output'][:100]}...")
print(f"\nAgent: {task_generate_recommendations['agent']}")


# ============================================================================
# **Question 3**: How many tasks have we defined in total for the multi-agent system? | **Answer**: You can use this cell to enter your answer. | Test individual agents
# ============================================================================

# Sample user data
sample_user_data = """
Restaurant Visit History:
- Visited "Spice Route" (Indian, $$) 5 times in the last 3 months
- Visited "Green Earth Cafe" (Vegan, $) 3 times
- Visited "Ramen House" (Japanese, $$) 2 times
- Visited "Taco Fiesta" (Mexican, $) 4 times

Social Media Posts:
- "Loving this spicy curry at Spice Route! 🌶️🔥"
- "Trying to eat more plant-based meals. This vegan bowl is delicious!"
- "Best ramen I've had in ages. The broth is perfection."
- "Late night tacos are the best tacos 🌮"
"""

def test_agent(agent_prompt: str, user_input: str) -> str:
    """Test an agent by sending it a sample input."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": agent_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

# Test the User Profile Generator
# Note: This will only work if you have set your Groq API key
try:
    profile_result = test_agent(
        agent_prompt=user_profile_prompt,
        user_input=f"Analyze this user data and create a profile:\n\n{sample_user_data}"
    )
    print("=" * 80)
    print("USER PROFILE GENERATED")
    print("=" * 80)
    print(profile_result)
except Exception as e:
    print(f"Note: Agent testing requires a valid Groq API key. Error: {e}")
    print("\nYou have successfully designed all agents and tasks.")
    print("In Lesson 2, you will integrate these agents into a working system.")


# ============================================================================
# **Question 4**: Based on the sample user data, which dietary preference is the user exploring? | **Answer**: You can use this cell to enter your answer. | Summary of agents and tasks
# ============================================================================

# Create a summary of all agents and their tasks
agents_summary = [
    {"agent": "User Profile Generator", "task": "Generate User Profile"},
    {"agent": "RAG Retriever", "task": "Retrieve Relevant Restaurants and Recipes"},
    {"agent": "Food Trend Analyst", "task": "Analyze Food Trends"},
    {"agent": "Food Style Expert", "task": "Analyze Food Styles"},
    {"agent": "Nutrition Expert", "task": "Evaluate Nutrition and Dietary Fit"},
    {"agent": "Recommendation Expert", "task": "Generate Final Recommendations"}
]

print("=" * 80)
print("MULTI-AGENT SYSTEM SUMMARY")
print("=" * 80)
for i, item in enumerate(agents_summary, 1):
    print(f"{i}. {item['agent']:30} → {item['task']}")

