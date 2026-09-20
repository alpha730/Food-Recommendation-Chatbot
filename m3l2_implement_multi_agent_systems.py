"""
M3L2 - Implement Multi-Agent Systems

Auto-extracted verbatim from: M3L2_Implement_Multi_Agent_Systems_2.ipynb
Code cells are unchanged; notebook magics are commented out.
"""


# ============================================================================
# Implement and Test a Multi-Agent Recommendation System | Learning Objectives | Introduction
# ============================================================================

# %%capture
# %pip install openai==1.99.9 | tail -n 1
# %pip install langchain==0.3.0
# %pip install langchain-openai==0.2.0
# %pip install langchain-community==0.3.0
# %pip install langgraph==0.2.0


# ============================================================================
# Import required libraries
# ============================================================================

import os
import json
from typing import TypedDict, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Swapped from OpenAI to Groq - see groq_client.py
# Set GROQ_API_KEY in the environment before running.
from groq_client import client, MODEL


# ============================================================================
# Import agent configurations
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
# Define the shared state structure
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
# **Question 1**: How many fields are in the AgentState structure? | **Answer**: You can use this cell to enter your answer. | Create the node functions
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


# ============================================================================
# Node 1: Generate the user profile
# ============================================================================

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


# ============================================================================
# Node 2: Retrieve the candidates
# ============================================================================

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


# ============================================================================
# Node 3: Analyze trends
# ============================================================================

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


# ============================================================================
# Node 4: Analyze food styles | **Task 1**: Complete the `node_analyze_styles` function
# ============================================================================

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


# ============================================================================
# Node 5: Evaluate nutrition
# ============================================================================

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


# ============================================================================
# Node 6: Generate recommendations
# ============================================================================

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
# **Question 2**: Which three nodes execute in parallel during Phase 3? | **Answer**: You can use this cell to enter your answer. | Build the workflow graph
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


# ============================================================================
# Test the multi-agent system | Test Case 1: The health-conscious user
# ============================================================================

test_user_1 = """
Restaurant Visit History:
- Visited "Green Bowl" (Vegan, $$) 8 times
- Visited "Mediterranean Grill" (Mediterranean, $$) 5 times
- Visited "Juice Lab" (Smoothies, $) 3 times

Social Media Posts:
- "Loving my plant-based journey! 🌱"
- "This gluten-free Mediterranean bowl is amazing!"
- "Fresh juice is the best way to start the day."

Dietary Restrictions: Vegan, Gluten-Free
"""

print("="*80)
print("TEST CASE 1: Health-Conscious User")
print("="*80)

try:
    result_1 = run_workflow(test_user_1)
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print(json.dumps(result_1["final_recommendations"], indent=2))
except Exception as e:
    print(f"\nTest requires valid Groq API key. Error: {e}")


# ============================================================================
# Test Case 2: The adventurous foodie
# ============================================================================

test_user_2 = """
Restaurant Visit History:
- Visited "Omakase Sushi" (Japanese Fine Dining, $$$$) 4 times
- Visited "Street Food Market" (International Fusion, $$) 6 times
- Visited "Molecular Gastronomy Lab" (Experimental, $$$$) 2 times

Social Media Posts:
- "Mind-blown by the 12-course tasting menu! 🤯"
- "Trying crickets for the first time. Surprisingly good!"
- "This molecular take on traditional ramen is art."

Dietary Restrictions: None
"""

print("="*80)
print("TEST CASE 2: Adventurous Foodie")
print("="*80)

try:
    result_2 = run_workflow(test_user_2)
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print(json.dumps(result_2["final_recommendations"], indent=2))
except Exception as e:
    print(f"\nTest requires valid Groq API key. Error: {e}")


# ============================================================================
# **Question 3**: Based on the workflow design, which phase takes the longest time to execute and why? | **Answer**: You can use this cell to enter your answer. | Evaluate the recommendations
# ============================================================================

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

# Evaluate Test Case 1 if available
try:
    if 'result_1' in locals():
        evaluate_recommendations(result_1)
except Exception as e:
    print(f"Evaluation requires completed test run: {e}")

