import streamlit as st
import sqlite3
import hashlib
import os
import json
from simple_rag_querier import SimpleRAGQuerier
from classify import GeminiBirdIdentifier
import torch
from datetime import datetime
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

torch.classes.__path__ = []  # add this line to manually set it to empty.

# --- CONFIG ---
PINECONE_INDEX_NAME = "cust"
EMBEDDING_MODEL = "intfloat/e5-large"
LLM_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = "AIzaSyA-uPJ6MseiNleEdhvpQON6vbAA-pIZ8VQ"

# --- DATABASE FUNCTIONS ---
def check_credentials(username, password):
    conn = sqlite3.connect("databases/users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result and result[0] == password

def insert_observation(name, common_name, coordinates, notes):
    conn = sqlite3.connect("databases/observations.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO observations (name, common_name, coordinates, notes)
        VALUES (?, ?, ?, ?)
    """, (name, common_name, coordinates, notes))
    conn.commit()
    conn.close()

# --- TASK CATEGORIES AND POINTS (INFORMATIONAL ONLY) ---
TASK_CATEGORIES = {
    "Leave No Trace": [
        ("Pack out all trash, including biodegradable items", 5),
        ("Pick up litter found on trail", 5),
        ("Sort or recycle trash properly", 10),
        ("Use eco-friendly toiletries", 3)
    ],
    "Observe & Document Biodiversity": [
        ("Record plant/animal sighting with date, location, and photo", 2),
        ("Upload observations to platforms like iNaturalist", 5),
        ("Note endangered or invasive species", 5)
    ],
    "Avoid Harmful Practices": [
        ("Stick to marked trails to avoid trampling vegetation", 5),
        ("Avoid disturbing wildlife (no feeding/chasing)", 5),
        ("Do not remove natural objects (rocks, flowers)", 5),
        ("Educate others on conservation principles", 10)
    ],
    "Contribute to Conservation": [
        ("Plant a native tree or help remove invasive plants", 15),
        ("Participate in a local conservation or trail maintenance activity", 15),
        ("Volunteer or donate to a local eco initiative", 10)
    ],
    "Eco-Aware Behavior": [
        ("Use a reusable water bottle/food container", 3),
        ("Use public or low-emission transportation to the trailhead", 5),
        ("Camp without leaving impact", 5),
        ("Maintain fully zero-waste day", 10)
    ],
    "Share & Educate": [
        ("Share a nature-related insight or eco-tip", 5),
        ("Teach someone about a plant or trail rule", 5),
        ("Create content promoting eco-hiking", 10)
    ],
    "Navigation & Preparedness": [
        ("Use maps and GPS instead of marking trees", 5),
        ("Respect local rules and protected areas", 5),
        ("Report trail damage or animal encounters to authorities", 10)
    ]
}

# Maximum points per category per day
MAX_DAILY_POINTS = {
    "Leave No Trace": 10,
    "Observe & Document Biodiversity": 10,
    "Avoid Harmful Practices": 10,
    "Contribute to Conservation": 15,
    "Eco-Aware Behavior": 10,
    "Share & Educate": 10,
    "Navigation & Preparedness": 10
}

# --- GEMINI BIRD IDENTIFIER ---
bird_identifier = GeminiBirdIdentifier(api_key=GEMINI_API_KEY)

# --- LOGIN PAGE ---
def login():
    st.title("🔐 RAG System Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid credentials")

def extract_species_locations(data):
    result = []
    for entry in data.get("species", []):
        name = entry.get("name", "Unknown")
        loc = entry.get("location", {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is not None and lon is not None:
            result.append((name, lat, lon))
    return result

# --- MAIN RAG APP ---
def rag_interface():
    st.title("📚 Simple RAG Q&A System")

    if "querier" not in st.session_state:
        st.session_state.querier = SimpleRAGQuerier(
            index_name=PINECONE_INDEX_NAME,
            embedding_model_name=EMBEDDING_MODEL,
            llm_model=LLM_MODEL,
            top_k=5
        )

    question = st.text_input("Ask your question:")

    if st.button("Submit") and question:
        with st.spinner("Fetching answer..."):
            answer = st.session_state.querier.query(question)
            st.session_state.answer = answer  # Save answer in session
            try:
                st.session_state.data = json.loads(answer)
            except Exception:
                st.session_state.data = None

    # If answer is stored, display it
    if "answer" in st.session_state:
        # Only display textual answer if it's NOT valid JSON (e.g., a plain answer)
        if not st.session_state.data:
            st.markdown("### ✅ Answer")
            st.write(st.session_state.answer)


    # Show Plot button only if valid data exists in session_state
    if "data" in st.session_state and st.session_state.data:
        data = st.session_state.data
        if "species" in data and isinstance(data["species"], list) and len(data["species"]) > 0:
            if st.button("📍 Plot on Map"):
                locations = extract_species_locations(data)
                print(locations)
                if locations:
                    heat_data = [[lat, lon] for _, lat, lon in locations]
                    avg_lat = sum(lat for _, lat, _ in locations) / len(locations)
                    avg_lon = sum(lon for _, _, lon in locations) / len(locations)

                    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
                    for name, lat, lon in locations:
                        folium.Marker([lat, lon], popup=name).add_to(m)
                    HeatMap(heat_data).add_to(m)

                    st.markdown("### 🗺️ Locations Map")
                    folium_static(m)
                else:
                    st.warning("No valid location data found in the JSON.")
        else:
            st.info("Answer JSON does not contain location data to plot.")

# --- OBSERVATION FORM ---
def observation_interface():
    st.title("📝 Add Wildlife Observation")
    
    # Initialize session state for observation points if not already present
    if 'observation_points' not in st.session_state:
        st.session_state.observation_points = 0
    
    # Display current observation points in sidebar
    st.sidebar.markdown("### 🔍 Observation Points")
    st.sidebar.markdown(f"**Current Points:** {st.session_state.observation_points}")
    
    # Reset observation points button
    if st.sidebar.button("Reset Observation Points"):
        st.session_state.observation_points = 0
        st.sidebar.success("Points reset to 0!")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    today_str = datetime.today().strftime('%Y-%m-%d')
    st.write(f"Date: {today_str} Records to be added in knowledge base ")
    coordinates = st.text_input("Coordinates")
    notes = st.text_area("Additional Notes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        get_suggestion = st.button("Get AI Suggestion")
    
    if uploaded_image and get_suggestion:
        with st.spinner("Identifying bird species..."):
            image_path = f"temp_{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            result_json = bird_identifier.identify_bird(image_path)
            print(result_json)
            os.remove(image_path)
        
        import re
        match = re.search(r"\{.*\}", result_json, re.DOTALL)
        if match:
            cleaned_json_str = match.group(0)
            try:
                result = json.loads(cleaned_json_str)
                name_info = result.get("scientific_name", {})
                print(name_info)
                if name_info:
                    common_name = result.get("common_name", "Unknown")
                    scientific_name = result.get("scientific_name", "Unknown")
                    confidence = result.get("classification_score", "N/A")
                    local_name = result.get("local_name", "Unknown")
                    st.markdown("### 🤖 AI Suggested Species")
                    st.info(
                        f"**Common Name:** {common_name}\n\n"
                        f"**Scientific Name:** *{scientific_name}*\n\n"
                        f"**Confidence Score:** {confidence}\n\n"
                        f"**Local Name:** *{local_name}*\n\n"
                    )
                    # Store in session to allow saving later
                    st.session_state.suggested_name = scientific_name
                    st.session_state.suggested_common = common_name
                else:
                    st.warning("No bird identification info returned by the model.")
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse JSON: {e}")
        else:
            st.error("No valid JSON found in the response.")
            
    # Conservation tasks related to this observation
    st.markdown("### 🌿 Conservation Tasks")
    st.write("Select the tasks you've completed with this observation:")
    
    # Create checkboxes for tasks related to biodiversity observation
    biodiversity_tasks = {
        "record_observation": ("Record plant/animal sighting with date, location, and photo", 2),
        "upload_platform": ("Upload observation to platforms like iNaturalist", 5),
        "note_endangered": ("Note if species is endangered or invasive", 5)
    }
    
    # Calculate points from selected tasks
    task_points = 0
    selected_tasks = {}
    
    for task_id, (task_desc, points) in biodiversity_tasks.items():
        selected = st.checkbox(f"{task_desc} (+{points} pts)")
        selected_tasks[task_id] = selected
        if selected:
            task_points += points
    
    # Save button with point calculation
    with col2:
        save_button = st.button("Save Observation & Points")
    
    if "suggested_name" in st.session_state and save_button:
        # Save to database
        insert_observation(
            st.session_state.suggested_name,
            st.session_state.suggested_common,
            coordinates,
            notes
        )
        
        # Update points
        st.session_state.observation_points += task_points
        
        # Show success messages
        st.success(f"Observation saved: {st.session_state.suggested_common} ({st.session_state.suggested_name})")
        
        if task_points > 0:
            st.balloons()
            st.success(f"🎉 You earned {task_points} points for your conservation actions!")
            st.sidebar.markdown(f"**Updated Points:** {st.session_state.observation_points}")
        
        # Clean up session state
        del st.session_state["suggested_name"]
        del st.session_state["suggested_common"]

# --- CONSERVATION TASKS TRACKING INTERFACE ---
def tasks_tracker_interface():
    st.title("🌿 Conservation Tasks Tracker")
    
    # Initialize session state for tracking selected tasks
    if 'selected_tasks' not in st.session_state:
        st.session_state.selected_tasks = {}
    
    st.write("""
    ### Track your conservation impact by checking completed tasks!
    
    Select the tasks you've completed today. Your score will be calculated automatically.
    """)
    
    # Track total score and category scores
    total_score = 0
    category_scores = {category: 0 for category in TASK_CATEGORIES.keys()}
    
    # Create a tab for each category
    tabs = st.tabs(list(TASK_CATEGORIES.keys()))
    
    for i, (category, tasks) in enumerate(TASK_CATEGORIES.items()):
        with tabs[i]:
            st.write(f"### {category}")
            st.write(f"Maximum daily points: **{MAX_DAILY_POINTS[category]}**")
            
            # Add checkboxes for each task in the category
            for task, points in tasks:
                task_id = f"{category}_{task}"
                if task_id not in st.session_state.selected_tasks:
                    st.session_state.selected_tasks[task_id] = False
                
                selected = st.checkbox(
                    f"{task} (+{points} pts)", 
                    key=task_id,
                    value=st.session_state.selected_tasks[task_id]
                )
                
                # Update the session state
                st.session_state.selected_tasks[task_id] = selected
                
                # Add points if selected
                if selected:
                    category_scores[category] += points
    
    # Apply daily maximums to each category
    capped_category_scores = {}
    for category, score in category_scores.items():
        capped_score = min(score, MAX_DAILY_POINTS[category])
        capped_category_scores[category] = capped_score
        total_score += capped_score
    
    # Display score summary
    st.sidebar.markdown("## 🏆 Your Score")
    
    # Create a dataframe for the scores
    import pandas as pd
    score_data = {
        "Category": [],
        "Your Points": [],
        "Max Points": [],
        "Capped Points": []
    }
    
    for category, score in category_scores.items():
        score_data["Category"].append(category)
        score_data["Your Points"].append(score)
        score_data["Max Points"].append(MAX_DAILY_POINTS[category])
        score_data["Capped Points"].append(capped_category_scores[category])
    
    score_df = pd.DataFrame(score_data)
    
    # Add the total row
    score_df.loc[len(score_df)] = [
        "Total", 
        sum(category_scores.values()), 
        sum(MAX_DAILY_POINTS.values()),
        total_score
    ]
    
    st.sidebar.dataframe(score_df, hide_index=True)
    
    # Display a progress bar for the total score
    max_possible = sum(MAX_DAILY_POINTS.values())
    progress = total_score / max_possible
    st.sidebar.progress(progress)
    
    # Add congratulatory message based on score
    if total_score > 0:
        if total_score < 20:
            st.sidebar.info("🌱 Good start! Keep going!")
        elif total_score < 40:
            st.sidebar.success("🌿 Great effort! You're making a difference!")
        else:
            st.sidebar.success("🌳 Amazing work! You're a conservation champion!")
    
    # Add a reset button
    if st.sidebar.button("Reset Tracker"):
        for key in st.session_state.selected_tasks:
            st.session_state.selected_tasks[key] = False
        st.experimental_rerun()

# --- CONSERVATION TASKS INFO INTERFACE ---
def tasks_info_interface():
    st.title("🌿 Nature Conservation Tasks")
    
    st.write("""
    ### Earn points by completing nature conservation tasks during your outdoor activities!
    
    This is an informational guide to help you track and assess your impact on nature while hiking or traveling.
    Complete these tasks to minimize your environmental footprint and contribute to conservation efforts.
    """)
    
    # Display daily max points by category
    st.subheader("Daily Maximum Points by Category")
    
    # Create a df for nicer display
    import pandas as pd
    max_points_data = {"Category": list(MAX_DAILY_POINTS.keys()), 
                       "Max Daily Points": list(MAX_DAILY_POINTS.values())}
    max_points_df = pd.DataFrame(max_points_data)
    max_points_df.loc[len(max_points_df)] = ["Total", sum(MAX_DAILY_POINTS.values())]
    st.table(max_points_df)
    
    # Display task categories and their point values
    st.subheader("Task Categories and Point Values")
    
    for category, tasks in TASK_CATEGORIES.items():
        with st.expander(f"🔍 {category}"):
            # Create a table for the tasks in this category
            task_data = {"Task": [], "Points": []}
            for task, points in tasks:
                task_data["Task"].append(task)
                task_data["Points"].append(points)
            
            # Convert to dataframe and display as table
            task_df = pd.DataFrame(task_data)
            st.table(task_df)
            
            # Add a description based on the category
            if category == "Leave No Trace":
                st.write("Leave No Trace principles are about minimizing your impact on the environment. Pack out all trash and leave natural areas as you found them.")
            elif category == "Observe & Document Biodiversity":
                st.write("Documenting biodiversity helps scientists track ecosystem health. Your observations contribute to conservation science.")
            elif category == "Avoid Harmful Practices":
                st.write("Preserving natural habitats means avoiding actions that can damage flora, fauna, or their environments.")
            elif category == "Contribute to Conservation":
                st.write("Direct conservation actions have the biggest impact. Getting involved with local initiatives amplifies your contribution.")
            elif category == "Eco-Aware Behavior":
                st.write("Sustainable choices reduce your environmental footprint while enjoying nature.")
            elif category == "Share & Educate":
                st.write("Spreading awareness helps create a community of conservation-minded individuals.")
            elif category == "Navigation & Preparedness":
                st.write("Being prepared and following rules helps protect both yourself and the environment.")
    
    # Add a section about how to use this information
    st.subheader("How to Use This System")
    st.write("""
    1. **Track Your Activities**: Use this guide to track your conservation activities
    2. **Self-Assessment**: At the end of each day, calculate your total points based on completed tasks
    3. **Set Goals**: Challenge yourself to reach a certain point threshold each day
    4. **Share Your Impact**: Inspire others by sharing your conservation efforts
    
    Remember, the real reward is knowing you're helping preserve natural spaces for future generations!
    """)

# --- MAIN APP SWITCHER ---
def main():
    menu = ["RAG Q&A", "Add Observation", "Conservation Tasks Tracker", "Conservation Tasks Guide"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Display user info in sidebar
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")

    if choice == "RAG Q&A":
        rag_interface()
    elif choice == "Add Observation":
        observation_interface()
    elif choice == "Conservation Tasks Tracker":
        tasks_tracker_interface()
    elif choice == "Conservation Tasks Guide":
        tasks_info_interface()
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# --- APP ENTRY POINT ---
if __name__ == "__main__":
    # Ensure database directories exist
    os.makedirs("databases", exist_ok=True)
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main()
    else:
        login()
