import streamlit as st

# Set wide layout
st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>Welcome to the Text Detector Main Landing Page! 👋</h1>", unsafe_allow_html=True)

# Define the button style with larger buttons and custom text sizes
button_style = """
<style>    
    .button-container {
        display: flex;
        justify-content: center; /* Center align all buttons */
        align-items: center; /* Vertically align buttons */
        gap: 30px; /* Specify spacing between buttons in pixels */
        padding: 20px; /* Optional padding around the container */
    }
    .rectangle-button {
        display: inline-block;
        width: 30%; /* Adjust width */
        height: 400px; /* Adjust height for multi-line text */
        background-color: #FF9800; /* Orange background */
        color: white; /* White text for button */
        text-align: center;
        line-height: 25px; /* Line height for better text spacing */
        padding: 10px;
        border-radius: 5px; /* Rounded corners */
        border: none; /* No border */
        cursor: pointer;
        text-decoration: none; /* No underline */
        font-size: 16px; /* Font size */
        overflow-wrap: break-word; /* Ensure long text wraps inside button */
    }
    .rectangle-button:hover {
        background-color: #FB8C00; /* Darker orange on hover */
    }
    .rectangle-button a {
        color: white; /* Ensure link text stays white */
        text-decoration: none; /* Remove any underline from links */
    }
    .sub-text {
        font-size: 15px; /* Smaller font size for sub-text */
        font-weight: normal; /* Optional: make it non-bold */
        padding-top: 10px; /* Add padding above the sub-text */
        padding-bottom: 5px; /* Add padding below the sub-text */
    }
    .detector-title {
        margin-top: 20px;
        font-size: 30px; /* Larger font size for "Profiles of People" */
        font-weight: bold; /* Ensure it stays bold */
        margin-bottom: 20px; /* Add space below "Profiles of People" */
    }
    .section-title {
        font-size: 20px; /* Slightly larger font size for section titles (Strengths/Limitations) */
        font-weight: bold;
    }
</style>
"""
# tab1, tab2 = st.tabs(["Single Text Evaluation", "Batch Text Evaluation"])
st.markdown("""
<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
        justify-content: center;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100%;
        display: flex;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    tab1, tab2 = st.tabs(["Single Text Evaluation", "Batch Text Evaluation"])
with tab1:
    # Add the style to Streamlit
    st.markdown(button_style, unsafe_allow_html=True)

    # Create a container with three custom buttons, each displaying detailed text
    st.markdown(
        """
        <div class="button-container">
            <a href="/Profiles_of_People_Single" class="rectangle-button">
                <div class="detector-title">Profiles of People
                <br>
                <span class="sub-text">Single Text Evaluation</span>
                </div>
            </a>
            <a href="/News_Articles_Single" class="rectangle-button">
                <div class="detector-title">News Articles
                <br>
                <span class="sub-text">Single Text Evaluation</span>
                </div>
            </a>
            <a href="/YouTube_Bot_Classifier" class="rectangle-button">
                <div class="detector-title">YouTube Bot Classifier
                <br>
                <span class="sub-text">Single Text Evaluation</span>
                </div>
            </a>            
        </div>
        """,
        unsafe_allow_html=True
    )

with tab2:
    # Add the style to Streamlit
    st.markdown(button_style, unsafe_allow_html=True)

    # Create a container with three custom buttons, each displaying detailed text
    st.markdown(
        """
        <div class="button-container">
            <a href="/Profiles_of_People_Batch" class="rectangle-button">
                <div class="detector-title">Profiles of People
                 <br>
                <span class="sub-text">Batch Text Evaluation</span>
                </div>
            </a>
            <a href="/News_Articles_Batch" class="rectangle-button">
                <div class="detector-title">News Articles
                <br>
                <span class="sub-text">Batch Text Evaluation</span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )