import streamlit as st
import base64
import os
from PIL import Image
from huggingface_hub import InferenceClient

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Hlal Chatbot",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Hlal Chatbot")

# =========================================
# SESSION INIT
# =========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_token" not in st.session_state:
    st.session_state.user_token = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================
# LOGIN SYSTEM
# =========================================
st.sidebar.title("🔐 Access Control")

if not st.session_state.authenticated:

    with st.sidebar.form("login_form"):
        user_api = st.text_input(
            "Enter your  API Key (optional)",
            type="password"
        )

        password = st.text_input(
            "enter password",
            type="password"
        )

        login_clicked = st.form_submit_button("Login")

    APP_PASSWORD = "Wael-1990"

    if login_clicked:

        # Direct API login
        if user_api:
            if user_api.startswith("hf_"):
                st.session_state.user_token = user_api
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.sidebar.error("Invalid API Key ❌")

        # Password login
        elif password == APP_PASSWORD:

            token = None

            try:
                if "HF_TOKEN" in st.secrets:
                    token = st.secrets["HF_TOKEN"]
            except Exception:
                pass

            if not token:
                token = os.getenv("HF_TOKEN")

            if token:
                st.session_state.user_token = token
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.sidebar.error("HF_TOKEN not configured ❌")

        else:
            st.sidebar.error("Invalid credentials ❌")

    st.stop()

# =========================================
# LOGOUT + CLEAR CHAT
# =========================================
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user_token = None
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# =========================================
# CREATE CLIENT
# =========================================
client = InferenceClient(token=st.session_state.user_token)

# =========================================
# SETTINGS SIDEBAR
# =========================================
st.sidebar.title("⚙ Settings")

mode = st.sidebar.radio(
    "Mode",
    ["Chat", "Image → Text", "Text → Image"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 200, 2000, 800)

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are Mr Hlal AI assistant. Respond clearly and professionally."
    )

with st.sidebar.form("system_prompt_form"):
    new_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=120
    )
    if st.form_submit_button("Apply"):
        st.session_state.system_prompt = new_prompt
        st.success("Updated ✅")

# =========================================
# WELCOME MESSAGE (ONLY ONCE)
# =========================================
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "You are welcome in Mr Hlal chatbot 👋\n\nHow can I assist you?"
    })

# =========================================
# CHAT MODE
# =========================================
if mode == "Chat":

    st.subheader("💬 Chat")

    user_input = st.chat_input("Type message and press Enter...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Generate response
    if user_input:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                stream = client.chat.completions.create(
                    model="Qwen/Qwen3.5-35B-A3B:novita",
                    messages=[
                        {"role": "system", "content": st.session_state.system_prompt}
                    ] + st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )

                for chunk in stream:
                    if (
                        chunk.choices
                        and len(chunk.choices) > 0
                        and hasattr(chunk.choices[0], "delta")
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                if full_response.strip():
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

            except Exception as e:
                st.error(str(e))

# =========================================
# IMAGE → TEXT
# =========================================
if mode == "Image → Text":

    st.subheader("🖼 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)

        if st.button("Generate Description"):

            with st.spinner("Analyzing..."):
                uploaded_file.seek(0)
                img_bytes = uploaded_file.read()
                base64_img = base64.b64encode(img_bytes).decode()

                prompt = f"""
                Analyze this image and describe it in detail.
                Image data:
                {base64_img[:4000]}
                """

                try:
                    response = client.chat.completions.create(
                        model="Qwen/Qwen3.5-35B-A3B:novita",
                        messages=[
                            {"role": "system", "content": st.session_state.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    st.success(response.choices[0].message.content)

                except Exception as e:
                    st.error(str(e))

# =========================================
# TEXT → IMAGE
# =========================================
if mode == "Text → Image":

    st.subheader("🎨 Generate Image")

    with st.form("image_form"):
        prompt = st.text_input("Enter prompt")
        submitted = st.form_submit_button("Generate")

        if submitted and prompt.strip():

            enhanced_prompt = f"""
            Ultra realistic high detail image of:
            {prompt}
            cinematic lighting, sharp focus, 8k, masterpiece
            """

            with st.spinner("Generating..."):
                try:
                    image = client.text_to_image(
                        enhanced_prompt,
                        model="stabilityai/stable-diffusion-xl-base-1.0"
                    )
                    st.image(image)

                except Exception as e:
                    st.error(str(e))