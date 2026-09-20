"""
Food Recommendation Agent - the complete application.

Assembled from the three Module 3 notebooks:

    M3L1_Design_Specialized_Agents_2.ipynb      -> the six agent configs + tasks
    M3L2_Implement_Multi_Agent_Systems_2.ipynb  -> shared state + workflow nodes
    M3L3_Build_Chatbot_Interface_2.ipynb        -> Gradio chat + database tabs

This build runs entirely on GROQ. Every OpenAI dependency from the notebooks
(the `openai` client, `langchain_openai.ChatOpenAI` and the OPENAI_API_KEY) has
been removed; the agent logic, prompts and interface are otherwise unchanged
from the notebooks.

The other new code is the "INTEGRATION LAYER" section near the end, which
finishes the agent by:

  * replacing Lesson 3's mock run_recommendation_workflow with a call to the
    real Lesson 2 multi-agent workflow (the notebook left a TODO saying exactly
    this: "In a real implementation, this would call the LangGraph workflow
    from Lesson 2"),
  * tolerating fenced JSON in model output, by wrapping call_agent rather than
    editing the notebook nodes,
  * adding a __main__ entry point.

Configuration lives in ONE place: the .env file next to groq_client.py.

    C:\Res\.env

        GROQ_API_KEY=gsk_your_key_here
        GROQ_MODEL=llama-3.3-70b-versatile      (optional)
        GROQ_CHAT_MODEL=llama-3.1-8b-instant    (optional)

Run:
    python food_agent.py
"""

import os
import json
import re
from typing import TypedDict, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr


# ============================================================================
# PART 1 - Groq client and models
#
# Nothing is configured here. The API key and model names live in ONE place,
# groq_client.py, which reads them from the .env file next to it. That module
# also replaces the notebooks' OpenAI client and ChatOpenAI instance; Groq's
# chat.completions API is call-compatible with the one the notebook nodes use,
# so nothing downstream had to change.
# ============================================================================

from groq_client import (
    client,
    MODEL,
    CHAT_MODEL,
    ChatGroq,
    SystemMessage,
    HumanMessage,
    GROQ_API_KEY,
    ENV_FILE,
)

llm = ChatGroq(model=CHAT_MODEL, temperature=0.7)


# ============================================================================
# PART 2 - M3L1: Specialized agent configurations
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
# PART 3 - M3L1: Agent prompts
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


# ============================================================================
# PART 4 - M3L1: Task definitions
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
# PART 5 - M3L1: Sample data, single-agent test, system summary
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


# ============================================================================
# PART 6 - M3L2: Agent configs used by the workflow
# ============================================================================

# Agent configurations from Lesson 1
agent_configs = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory": """You are an expert user behavior analyst with 10 years of experience in the food industry. 
        You excel at identifying patterns in dining behavior and building rich user profiles."""
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search."""
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends and emerging dining concepts.",
        "backstory": """You are a culinary journalist who has spent 15 years covering food culture across global markets."""
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines."""
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": """You are a registered dietitian with 8 years of clinical experience."""
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents into final recommendations.",
        "backstory": """You are a recommendation systems architect with experience in personalization engines."""
    }
}

print("Agent configurations loaded successfully!")


# ============================================================================
# PART 7 - M3L2: Shared state
# ============================================================================

# Define the shared state structure as a dictionary.
# Every node reads from and writes to this state.
INITIAL_STATE = {
    # Input
    "user_input": "",
    
    # Phase 1: User Analysis
    "user_profile": {},
    
    # Phase 2: Data Retrieval
    "retrieved_restaurants": [],
    "retrieved_recipes": [],
    
    # Phase 3: Analysis (Parallel)
    "trend_analysis": {},
    "style_analysis": {},
    "nutrition_analysis": {},
    
    # Phase 4: Synthesis
    "final_recommendations": {},
    
    # Metadata
    "workflow_step": "start"
}

print(f"State structure defined with {len(INITIAL_STATE)} fields:")
for key in INITIAL_STATE:
    print(f"  - {key}")


# ============================================================================
# PART 8 - M3L2: Agent caller and workflow nodes
# ============================================================================

def call_agent(agent_key: str, user_message: str) -> str:
    """Call an agent with a specific message and return its response."""
    config = agent_configs[agent_key]
    
    system_prompt = f"""You are a {config['role']}.
    
Your goal: {config['goal']}

Your background: {config['backstory']}

Respond with structured, actionable output."""
    
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

def node_generate_profile(state: dict) -> dict:
    """Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")
    
    user_message = f"""Analyze this user data and create a comprehensive profile:

{state['user_input']}

Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- summary (string)
"""
    
    try:
        response = call_agent("user_profile_generator", user_message)
        user_profile = json.loads(response)
        print(f"✓ User profile generated: {user_profile.get('summary', 'No summary')}")
    except Exception as e:
        print(f"⚠ Error generating profile: {e}")
        user_profile = {"error": str(e)}
    
    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state

def node_retrieve_candidates(state: dict) -> dict:
    """Retrieve restaurant and recipe candidates from vector database."""
    print("\n[Phase 2] Retrieving candidates from vector database...")
    
    profile = state["user_profile"]
    
    user_message = f"""Based on this user profile:
{json.dumps(profile, indent=2)}

Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.

Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes: [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]

Make the results realistic and diverse.
"""
    
    try:
        response = call_agent("rag_retriever", user_message)
        retrieved_data = json.loads(response)
        restaurants = retrieved_data.get("restaurants", [])
        recipes = retrieved_data.get("recipes", [])
        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []
    
    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state

def node_analyze_trends(state: dict) -> dict:
    """Analyze food trends in the retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    
    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"✓ Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"⚠ Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}
    
    state["trend_analysis"] = trend_analysis
    return state

def node_analyze_styles(state: dict) -> dict:
    """Analyze food styles and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Analyze the culinary styles and flavor profiles of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

For each item, identify the cuisine type and its regional variation, the dominant cooking
methods and techniques, the core flavor profile (spicy, umami-rich, bright and acidic,
rich and creamy, smoky, sweet), and whether it is traditional, modern, or fusion.
Then assess how well each style matches the user's flavor preferences and adventurousness
score, giving a short culinary justification for every judgement.

Return JSON:
{{"style_analysis": [{{"name": str, "cuisine_type": str, "regional_variation": str,
"cooking_methods": [str], "flavor_profile": [str], "traditional_or_fusion": str,
"match_score": int, "justification": str}}],
"strong_matches": [str], "stretch_options": [str], "poor_matches": [str],
"summary": str}}
"""  # Fill this in
    
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print(f"✓ Style analysis completed")
    except Exception as e:
        print(f"⚠ Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}
    
    state["style_analysis"] = style_analysis
    return state

def node_evaluate_nutrition(state: dict) -> dict:
    """Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print(f"✓ Nutrition evaluation completed")
    except Exception as e:
        print(f"⚠ Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}
    
    state["nutrition_analysis"] = nutrition_analysis
    return state

def node_generate_recommendations(state: dict) -> dict:
    """Synthesize all analyses into final recommendations."""
    print("\n[Phase 4] Generating final recommendations...")
    
    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}

Return JSON:
{{
  "restaurants": [{{"name": str, "reasoning": str}}],
  "recipes": [{{"name": str, "reasoning": str}}]
}}

Each reasoning should be 2-3 sentences explaining why it's a great match.
"""
    
    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"✓ Generated {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"✓ Generated {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"⚠ Error generating recommendations: {e}")
        recommendations = {"error": str(e)}
    
    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return state


# ============================================================================
# PART 9 - M3L2: The workflow and its evaluation
# ============================================================================

def run_workflow(user_input: str) -> dict:
    """Run the full multi-agent workflow.
    
    Phases:
      1. User Analysis      (sequential)
      2. Data Retrieval      (sequential)
      3. Analysis            (parallel – trends, styles, nutrition)
      4. Synthesis           (sequential)
    """
    
    # Initialize shared state
    state = {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start"
    }
    
    # Phase 1 – Sequential
    state = node_generate_profile(state)
    
    # Phase 2 – Sequential
    state = node_retrieve_candidates(state)
    
    # Phase 3 – Parallel using ThreadPoolExecutor
    print("\n[Phase 3] Running analysis agents in parallel...")
    
    # Each function needs its own copy of state to read from,
    # and we merge their outputs back afterwards.
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends   = executor.submit(node_analyze_trends, dict(state))
        future_styles   = executor.submit(node_analyze_styles, dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))
        
        result_trends   = future_trends.result()
        result_styles   = future_styles.result()
        result_nutrition = future_nutrition.result()
        
    # Merge parallel results back into state
    state["trend_analysis"]    = result_trends["trend_analysis"]
    state["style_analysis"]    = result_styles["style_analysis"]
    state["nutrition_analysis"] = result_nutrition["nutrition_analysis"]
    
    # Phase 4 – Sequential
    state = node_generate_recommendations(state)
    
    return state

print("✓ Workflow function built successfully!")

def evaluate_recommendations(result: Dict[str, Any]):
    """Evaluate the quality of recommendations."""
    print("\n" + "="*80)
    print("RECOMMENDATION EVALUATION")
    print("="*80)
    
    profile = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})
    
    # Check if recommendations exist
    restaurants = recommendations.get("restaurants", [])
    recipes = recommendations.get("recipes", [])
    
    print(f"\n✓ Number of restaurant recommendations: {len(restaurants)}")
    print(f"✓ Number of recipe recommendations: {len(recipes)}")
    
    # Check dietary compliance
    dietary_restrictions = profile.get("dietary_restrictions", [])
    if dietary_restrictions:
        print(f"\n✓ Dietary restrictions identified: {', '.join(dietary_restrictions)}")
        print("  → Check if recommendations respect these restrictions")
    
    # Check diversity
    favorite_cuisines = profile.get("favorite_cuisines", [])
    if favorite_cuisines:
        print(f"\n✓ Favorite cuisines: {', '.join(favorite_cuisines)}")
        print("  → Check if recommendations include these cuisines")
    
    # Evaluate reasoning quality
    if restaurants:
        print(f"\n✓ First restaurant recommendation:")
        print(f"  Name: {restaurants[0].get('name', 'N/A')}")
        print(f"  Reasoning: {restaurants[0].get('reasoning', 'N/A')}")
    
    if recipes:
        print(f"\n✓ First recipe recommendation:")
        print(f"  Name: {recipes[0].get('name', 'N/A')}")
        print(f"  Reasoning: {recipes[0].get('reasoning', 'N/A')}")
    
    print("\n" + "="*80)


# ============================================================================
# PART 10 - M3L3: Intent classification and preference extraction
# ============================================================================

def classify_intent(user_message: str, llm: ChatGroq) -> str:
    """Classify user intent as restaurant, recipe, both, or clarification."""
    
    system_prompt = """You are an intent classifier for a food recommendation system.
    
Analyze the user's message and classify it as ONE of:
- "restaurant" - User wants restaurant recommendations
- "recipe" - User wants recipe recommendations
- "both" - User wants both restaurant and recipe recommendations
- "clarification" - User needs help or is asking a question
- "database" - User wants to add/edit/delete database entries

Examples:
"Where should I eat tonight?" → restaurant
"How do I make lasagna?" → recipe
"I want dinner ideas" → both
"What can you help me with?" → clarification
"I want to add a new restaurant" → database

Respond with ONLY the classification label."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    intent = response.content.strip().lower()
    
    # Validate intent
    valid_intents = ["restaurant", "recipe", "both", "clarification", "database"]
    if intent not in valid_intents:
        intent = "clarification"
    
    return intent

print("Intent classification function created!")

def extract_preferences(user_message: str, llm: ChatGroq) -> Dict[str, Any]:
    """Extract user preferences from natural language input."""
    
    system_prompt = """You are a preference extractor for a food recommendation system.
    
Extract user preferences from their message and return JSON with these keys:
- favorite_cuisines: List of mentioned cuisines (e.g., ["Italian", "Thai"])
- dietary_restrictions: List of dietary needs (e.g., ["vegetarian", "gluten-free"])
- dining_occasion: Type of dining (e.g., "casual", "fine dining", "quick bite")
- price_range: Price preference (e.g., "$", "$$", "$$$", "$$$$")
- flavor_preferences: List of flavor preferences (e.g., ["spicy", "sweet"])
- other_preferences: Any other relevant details

If a field is not mentioned, use an empty list or "not specified".

Example:
Input: "I love spicy Thai food and I'm vegetarian"
Output: {
  "favorite_cuisines": ["Thai"],
  "dietary_restrictions": ["vegetarian"],
  "dining_occasion": "not specified",
  "price_range": "not specified",
  "flavor_preferences": ["spicy"],
  "other_preferences": ""
}

Respond with ONLY valid JSON."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    
    try:
        preferences = json.loads(response.content)
    except:
        # Fallback if parsing fails
        preferences = {
            "favorite_cuisines": [],
            "dietary_restrictions": [],
            "dining_occasion": "not specified",
            "price_range": "not specified",
            "flavor_preferences": [],
            "other_preferences": ""
        }
    
    return preferences

print("Preference extraction function created!")


# ============================================================================
# PART 11 - M3L3: Formatting and database management
# ============================================================================

def format_recommendations(recommendations: Dict[str, Any]) -> str:
    """Format recommendations for display in the chat."""
    
    output = ""
    
    # Format restaurant recommendations
    if "restaurants" in recommendations and recommendations["restaurants"]:
        output += "🍽️ **Restaurant Recommendations:**\n\n"
        for i, restaurant in enumerate(recommendations["restaurants"], 1):
            output += f"**{i}. {restaurant['name']}**\n"
            output += f"   - Cuisine: {restaurant['cuisine']}\n"
            output += f"   - Price: {restaurant['price']}\n"
            output += f"   - Why: {restaurant['reasoning']}\n\n"
    
    # Format recipe recommendations
    if "recipes" in recommendations and recommendations["recipes"]:
        output += "👨‍🍳 **Recipe Recommendations:**\n\n"
        for i, recipe in enumerate(recommendations["recipes"], 1):
            output += f"**{i}. {recipe['name']}**\n"
            output += f"   - Cuisine: {recipe['cuisine']}\n"
            output += f"   - Difficulty: {recipe['difficulty']}\n"
            output += f"   - Why: {recipe['reasoning']}\n\n"
    
    if not output:
        output = "I couldn't generate recommendations. Please try again with more details about your preferences."
    
    return output

print("Formatting function created!")

def add_restaurant(name: str, cuisine: str, price: str, location: str, description: str) -> str:
    """Add a new restaurant to the database."""
    # In a real implementation, this would add to the vector database
    print(f"Adding restaurant: {name}")
    return f"✅ Successfully added '{name}' to the database!"

def add_recipe(name: str, cuisine: str, difficulty: str, prep_time: str, ingredients: str, instructions: str) -> str:
    """Add a new recipe to the database."""
    # In a real implementation, this would add to the vector database
    print(f"Adding recipe: {name}")
    return f"✅ Successfully added '{name}' recipe to the database!"

print("Database management functions created!")


# ============================================================================
# PART 12 - M3L3: The chatbot function
# ============================================================================

def recommendation_chatbot(message: str, history: List[Tuple[str, str]]) -> str:
    """Main chatbot function that handles user requests."""
    
    try:
        # Step 1: Classify intent
        intent = classify_intent(message, llm)
        print(f"Classified intent: {intent}")
        
        # Step 2: Handle different intents
        if intent == "clarification":
            return """I'm your food recommendation assistant! I can help you with:
            
🍽️ **Restaurant recommendations** - Tell me your cuisine preferences, dietary restrictions, and occasion
👨‍🍳 **Recipe recommendations** - Let me know what you'd like to cook
📝 **Database management** - Add, update, or delete restaurants and recipes

Just describe what you're looking for, and I'll provide personalized recommendations!"""
        
        elif intent == "database":
            return """To manage the database, please use the tabs above:
            
- **Add Restaurant**: Submit a new restaurant
- **Add Recipe**: Submit a new recipe
- **Edit/Delete**: Modify or remove existing entries

Is there anything else I can help you with?"""
        
        elif intent in ["restaurant", "recipe", "both"]:
            # Step 3: Extract preferences
            preferences = extract_preferences(message, llm)
            print(f"Extracted preferences: {preferences}")
            
            # Step 4: Run workflow
            recommendations = run_recommendation_workflow(preferences, intent)
            
            # Step 5: Format output
            formatted_output = format_recommendations(recommendations)
            
            return formatted_output
        
        else:
            return "I'm not sure how to help with that. Can you rephrase your request?"
    
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please make sure you have set your Groq API key."

print("Complete chatbot function created!")


# ============================================================================
# INTEGRATION LAYER (new code - not from the notebooks)
#
# Lesson 3 shipped run_recommendation_workflow with hard-coded mock data and
# a TODO to call the Lesson 2 workflow. This section does that. Nothing above
# this line was modified.
# ============================================================================

def _strip_json_fences(text: str) -> str:
    """Return the JSON payload from a model reply that may be fenced.

    The Lesson 2 nodes call json.loads() straight on the model output, and
    models often wrap JSON in code fences, which makes that call fail. Rather
    than edit the notebook nodes, we wrap call_agent so they always receive
    clean JSON text.
    """
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.S)
    if fence:
        return fence.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start:end + 1]
    return stripped


_notebook_call_agent = call_agent


def call_agent(agent_key: str, user_message: str) -> str:  # noqa: F811
    """call_agent from Lesson 2, made tolerant of fenced JSON output."""
    return _strip_json_fences(_notebook_call_agent(agent_key, user_message))


def preferences_to_user_input(preferences: Dict[str, Any]) -> str:
    """Turn the preferences extracted in chat into the free-text block that the
    Lesson 2 workflow expects as user_input."""
    def as_text(value):
        if isinstance(value, list):
            return ", ".join(str(v) for v in value) if value else "not specified"
        return str(value) if value else "not specified"

    return f"""Stated Preferences (from chat):
- Favorite cuisines: {as_text(preferences.get('favorite_cuisines'))}
- Dietary restrictions: {as_text(preferences.get('dietary_restrictions'))}
- Dining occasion: {as_text(preferences.get('dining_occasion'))}
- Price range: {as_text(preferences.get('price_range'))}
- Flavor preferences: {as_text(preferences.get('flavor_preferences'))}
- Other notes: {as_text(preferences.get('other_preferences'))}
"""


def run_recommendation_workflow(preferences: Dict[str, Any],
                                recommendation_type: str) -> Dict[str, Any]:
    """Run the real multi-agent workflow and return recommendations.

    Args:
        preferences: User preferences extracted from their message
        recommendation_type: "restaurant", "recipe", or "both"

    Returns:
        Dictionary with recommendations, shaped for format_recommendations()
    """
    print(f"Running workflow for {recommendation_type} recommendations...")

    state = run_workflow(preferences_to_user_input(preferences))
    result = state.get("final_recommendations", {}) or {}

    if "error" in result:
        return {}

    # format_recommendations() expects cuisine/price/difficulty on each item,
    # but the synthesis agent returns name/reasoning only, so fill the display
    # fields from the retrieved candidates wherever the names line up.
    retrieved = {item.get("name"): item
                 for item in (state.get("retrieved_restaurants") or [])
                 + (state.get("retrieved_recipes") or [])
                 if isinstance(item, dict)}

    def decorate(items, extra_key, extra_default):
        decorated = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            source = retrieved.get(item.get("name"), {})
            decorated.append({
                "name": item.get("name", "Unnamed"),
                "cuisine": item.get("cuisine") or source.get("cuisine", "Not specified"),
                extra_key: item.get(extra_key) or source.get(extra_key, extra_default),
                "reasoning": item.get("reasoning", "No reasoning provided."),
            })
        return decorated

    restaurants = decorate(result.get("restaurants"), "price", "Not specified")
    recipes = decorate(result.get("recipes"), "difficulty", "Not specified")

    if recommendation_type == "restaurant":
        return {"restaurants": restaurants}
    elif recommendation_type == "recipe":
        return {"recipes": recipes}
    else:  # both
        return {"restaurants": restaurants, "recipes": recipes}


# ============================================================================
# PART 13 - M3L3: The Gradio interface
# ============================================================================

# Create the main interface with tabs
with gr.Blocks(title="Food Recommendation Chatbot", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🍽️ Food Recommendation Chatbot
    
    Your personal AI assistant for restaurant and recipe recommendations!
    """)
    
    with gr.Tabs():
        
        # Tab 1: Chat Interface
        with gr.Tab("💬 Chat"):
            chatbot_interface = gr.ChatInterface(
                fn=recommendation_chatbot,
                examples=[
                    "I'm looking for vegetarian restaurants",
                    "Suggest some easy recipes for dinner",
                    "I want spicy Thai food recommendations",
                    "What can you help me with?"
                ],
                title="Chat with the Recommendation Assistant",
                description="Describe your food preferences and I'll recommend restaurants or recipes!"
            )
        
        # Tab 2: Add Restaurant
        with gr.Tab("➕ Add Restaurant"):
            gr.Markdown("### Add a New Restaurant to the Database")
            
            with gr.Row():
                with gr.Column():
                    rest_name = gr.Textbox(label="Restaurant Name")
                    rest_cuisine = gr.Textbox(label="Cuisine Type")
                    rest_price = gr.Dropdown(
                        choices=["$", "$$", "$$$", "$$$$"],
                        label="Price Range"
                    )
                with gr.Column():
                    rest_location = gr.Textbox(label="Location")
                    rest_description = gr.Textbox(
                        label="Description",
                        lines=3
                    )
            
            add_rest_btn = gr.Button("Add Restaurant", variant="primary")
            rest_output = gr.Textbox(label="Status")
            
            add_rest_btn.click(
                fn=add_restaurant,
                inputs=[rest_name, rest_cuisine, rest_price, rest_location, rest_description],
                outputs=rest_output
            )
        
        # Tab 3: Add Recipe
        with gr.Tab("➕ Add Recipe"):
            gr.Markdown("### Add a New Recipe to the Database")
            
            with gr.Row():
                with gr.Column():
                    recipe_name = gr.Textbox(label="Recipe Name")
                    recipe_cuisine = gr.Textbox(label="Cuisine Type")
                    recipe_difficulty = gr.Dropdown(
                        choices=["Easy", "Medium", "Hard"],
                        label="Difficulty"
                    )
                with gr.Column():
                    recipe_time = gr.Textbox(label="Prep Time")
                    recipe_ingredients = gr.Textbox(
                        label="Ingredients (comma-separated)",
                        lines=3
                    )
            
            recipe_instructions = gr.Textbox(
                label="Instructions",
                lines=5
            )
            
            add_recipe_btn = gr.Button("Add Recipe", variant="primary")
            recipe_output = gr.Textbox(label="Status")
            
            add_recipe_btn.click(
                fn=add_recipe,
                inputs=[recipe_name, recipe_cuisine, recipe_difficulty, recipe_time, recipe_ingredients, recipe_instructions],
                outputs=recipe_output
            )
        
        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Chatbot
            
            This chatbot uses a multi-agent AI system to provide personalized food recommendations.
            
            ### Features:
            - 🤖 **Intelligent Agents**: Six specialized AI agents work together to analyze your preferences
            - 🔍 **Smart Search**: Vector database retrieval finds the most relevant options
            - 🎯 **Personalized**: Recommendations tailored to your tastes and dietary needs
            - 📝 **Editable Database**: Add your favorite restaurants and recipes
            
            ### How to Use:
            1. Go to the **Chat** tab
            2. Describe what you're looking for (cuisine, dietary restrictions, occasion, etc.)
            3. Receive personalized restaurant or recipe recommendations
            4. Use the **Add** tabs to contribute to the database
            
            ### Technologies:
            - A hand-rolled multi-agent workflow on the Groq API
            - Groq (Llama 3.3) for language understanding
            - Vector databases for semantic search
            - Gradio for the user interface
            """)

print("Complete interface created!")
print("\nTo launch the chatbot, run: demo.launch()")


# ============================================================================
# Entry point (new code - the notebook used demo.launch(share=True))
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-AGENT SYSTEM SUMMARY")
    print("=" * 80)
    for i, item in enumerate(agents_summary, 1):
        print(f"{i}. {item['agent']:30} -> {item['task']}")

    if not os.environ.get("GROQ_API_KEY"):
        print("\nWarning: GROQ_API_KEY is not set - the agents will fail to run.")

    demo.launch(share=True)
