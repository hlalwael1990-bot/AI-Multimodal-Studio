import streamlit as st
import base64
import os
from PIL import Image
from huggingface_hub import InferenceClient

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Multimodal Studio",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Multimodal Studio")

# =========================================
# ACCESS CONTROL (API OR PASSWORD)
# =========================================
st.sidebar.title("🔐 Access Control")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_token" not in st.session_state:
    st.session_state.user_token = None

if not st.session_state.authenticated:

    user_api = st.sidebar.text_input(
        "Enter your HuggingFace API Key (optional)",
        type="password"
    )

    password = st.sidebar.text_input(
        "Or enter password",
        type="password"
    )

    # يمكنك تغيير كلمة المرور هنا
    APP_PASSWORD = "WaelAI1990"

    if user_api:
        if user_api.startswith("hf_"):
            st.session_state.user_token = user_api
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.sidebar.error("Invalid API Key ❌")

    elif password == APP_PASSWORD:
        # يعمل على Cloud أو Local
        token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

        if token:
            st.session_state.user_token = token
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.sidebar.error("HF_TOKEN not configured in Secrets ❌")

    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user_token = None
    st.rerun()

# =========================================
# CREATE CLIENT
# =========================================
client = InferenceClient(token=st.session_state.user_token)

# =========================================
# SETTINGS
# =========================================
st.sidebar.title("⚙ Settings")

mode = st.sidebar.radio(
    "Mode",
    ["Image → Text", "Text → Image", "Chat"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 200, 2000, 800)

# =========================================
# SYSTEM PROMPT
# =========================================
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are Qwen 3.5 35B advanced assistant. "
        "Respond clearly and professionally."
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

        zoom = st.slider("Zoom", 0.5, 3.0, 1.0, 0.1)
        w, h = image.size
        resized = image.resize((int(w * zoom), int(h * zoom)))
        st.image(resized)

        if st.button("Generate Description"):

            with st.spinner("Analyzing..."):

                uploaded_file.seek(0)
                img_bytes = uploaded_file.read()
                base64_img = base64.b64encode(img_bytes).decode()

                prompt = f"""
                Analyze this image content and describe in detail.
                Image data:
                {base64_img[:4000]}
                """

                try:
                    response = client.chat.completions.create(
                        model="Qwen/Qwen3.5-35B-A3B:novita",
                        messages=[
                            {
                                "role": "system",
                                "content": st.session_state.system_prompt
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
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
        prompt = st.text_input("Enter prompt and press Enter")
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

# =========================================
# CHAT (STREAMING)
# =========================================
if mode == "Chat":

    st.subheader("💬 Chat (Streaming)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type message and press Enter...")

    if user_input:

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):

            message_placeholder = st.empty()
            full_response = ""

            try:
                stream = client.chat.completions.create(
                    model="Qwen/Qwen3.5-35B-A3B:novita",
                    messages=[
                        {
                            "role": "system",
                            "content": st.session_state.system_prompt
                        }
                    ] + st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(str(e))