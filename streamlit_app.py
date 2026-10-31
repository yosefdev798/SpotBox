import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests


# ============================================================
# 1. GROQ LLM FUNCTION
# ============================================================

def ask_llm(prompt, context):

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        return "❌ API key not found. Please set GROQ_API_KEY in your system environment."

    # Correct Groq API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-20b",


        "messages": [
            {
                "role": "system",
                "content": (
                    "You are SpotBox, an AI assistant that understands "
                    "objects detected by a computer vision system. "
                    f"The currently detected objects are: {context}. "
                    "Answer the user's question using this information."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0.5
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=data
        )

        if response.status_code == 200:

            result = response.json()

            return result["choices"][0]["message"]["content"]

        else:

            return (
                f"❌ Error from Groq API: "
                f"{response.status_code} - {response.text}"
            )

    except Exception as e:

        return f"❌ Request failed: {str(e)}"


# ============================================================
# 2. STREAMLIT PAGE
# ============================================================

st.title("🚀 SpotBox - Intelligent Object Detection & Chat")

st.write(
    "Upload an image to detect objects and "
    "talk with SpotBox AI."
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "detected_labels" not in st.session_state:
    st.session_state.detected_labels = []


# ============================================================
# 4. LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_yolo_model():

    return YOLO("yolov8n.pt")


model = load_yolo_model()


# ============================================================
# 5. IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# 6. YOLO OBJECT DETECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Source Image"
    )

    st.write("🔍 Detecting objects...")

    results = model(image)

    current_labels = []


    for result in results:

        # Draw bounding boxes
        im_array = result.plot()

        im = Image.fromarray(
            im_array[..., ::-1]
        )

        st.image(
            im,
            caption="Processed Detections Output"
        )


        # Extract detected object names
        for box in result.boxes:

            class_id = int(box.cls)

            label_name = model.names[class_id]

            current_labels.append(label_name)


    # Save detected objects
    st.session_state.detected_labels = current_labels


# ============================================================
# 7. CHAT INTERFACE
# ============================================================

st.markdown("---")

st.subheader(
    "💬 Converse with SpotBox Visual Intelligence"
)


# Display previous conversation
for interaction in st.session_state.chat_history:

    with st.chat_message(interaction["role"]):

        st.markdown(
            interaction["content"]
        )


# ============================================================
# 8. USER QUESTION
# ============================================================

if user_prompt := st.chat_input(
    "Ask me something about the detected objects..."
):

    # Display user message
    with st.chat_message("user"):

        st.markdown(user_prompt)


    # Save user message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )


    # ========================================================
    # 9. BUILD VISION CONTEXT
    # ========================================================

    if st.session_state.detected_labels:

        visual_context_str = ", ".join(
            st.session_state.detected_labels
        )

    else:

        visual_context_str = (
            "No objects are currently detected."
        )


    # ========================================================
    # 10. ASK GROQ / LLAMA
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "SpotBox is thinking..."
        ):

            ai_response = ask_llm(
                prompt=user_prompt,
                context=visual_context_str
            )

            st.markdown(ai_response)


    # ========================================================
    # 11. SAVE AI RESPONSE
    # ========================================================

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )
