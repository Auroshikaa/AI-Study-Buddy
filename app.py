import streamlit as st
st.set_page_config(page_title="Agentic Study Assistant", layout="wide")
import re
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()

# Langchain setup
import os
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found. Add it to Streamlit Secrets or .env file.")
    st.stop()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2, model_name="gpt-3.5-turbo")


search = DuckDuckGoSearchRun()

# Prompts
planner_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Break the topic '{topic}' into 4–6 subtopics with short explanations."
)
researcher_prompt = PromptTemplate(
    input_variables=["query"],
    template="Extract the most useful knowledge for study from:\n\n{query}"
)
summarizer_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the content below into bullet-pointed notes with **bolded** key terms:\n\n{text}"
)
quizzer_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Based on these notes:
    {text}
    Create 5 multiple-choice questions with four options each.
    Format:
    1. Question?
    a) Option1
    b) Option2
    c) Option3
    d) Option4
    Answer: b)
    Explanation: ...
    """
)
suggestion_prompt = PromptTemplate(
    input_variables=["topic", "score"],
    template="Suggest 2-3 next concepts based on topic '{topic}' and score {score}/5."
)

# Chains
planner_chain = LLMChain(llm=llm, prompt=planner_prompt)
researcher_chain = LLMChain(llm=llm, prompt=researcher_prompt)
summarizer_chain = LLMChain(llm=llm, prompt=summarizer_prompt)
quizzer_chain = LLMChain(llm=llm, prompt=quizzer_prompt)
suggestion_chain = LLMChain(llm=llm, prompt=suggestion_prompt)

# --- UI CONFIG ---
#st.set_page_config(page_title="Agentic Study Assistant", layout="wide")

# --- Session State ---
for key, default in {
    "dark_mode": False,
    "learning_log": [],
    "saved_notes": {},
    "current_tab": "Home",
    "input_mode": "Topic",
    "summary": "",
    "quiz": [],
    "correct_answers": [],
    "explanations": [],
    "user_answers": [],
    "submitted": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Custom CSS ---
def inject_custom_css(dark_mode: bool):
    if dark_mode:
        st.markdown("""
            <style>
                body, .stApp { background-color: #1e1e1e; color: #f1e7fe; }

                header[data-testid="stHeader"] {
                    background-color: #101010; border-bottom: 1px solid black;
                }

                .stSidebar { background-color: #151515 !important; border-right: 1px solid black; }

                .stTextInput>div>div>input {
                    background-color: #2c2c2c; border: 1px solid #b28dff; color: #f3f3f3;
                }

                .stTextInput>label,
                .stSidebar label,
                .stSidebar section,
                .stSidebar div,
                label[data-testid="stMarkdownContainer"],
                .stRadio label,
                .stRadio div,
                div[role="radiogroup"] > div > label,
                div[role="radiogroup"] > label {
                    color: #e0bbff !important;
                }

                .stButton>button {
                    background-color: #2c2c2c;
                    color: #e0bbff;
                    border: 1px solid #a78bfa;
                    border-radius: 6px;
                }

                .stButton>button:hover { background-color: #3d3d3d; }

                h1, h2, h3, h4 { color: #f8f8ff; }

                .stFileUploader {
                    background-color: #2c2c2c !important;
                    border: 1px dashed #a78bfa !important;
                    border-radius: 8px;
                    color: #e0bbff;
                }
                    /* Quiz font styling */
                div[data-testid="stRadio"] > label {
                    font-size: 1.2rem !important;  /* Question font size */
                    font-weight: 600 !important;
                    margin-bottom: 0.5rem;
                }
                div[role="radiogroup"] > div > label {
                    font-size: 0.9rem !important;  /* Answer option font size */
                    font-weight: 400 !important;
                }

            </style>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <style>
                body, .stApp { background-color: #f3f0ff; color: #222; }
                header[data-testid="stHeader"] {
                    background-color: #e5e1ff; border-bottom: 1px solid #dcd0ff;
                }
                .stSidebar { background-color: #eaeaff; border-right: 1px solid #dcd0ff; }
                .stTextInput>div>div>input {
                    background-color: #ffffff; border: 1px solid #9370db; color: #111;
                }
                .stTextInput>label, .stSidebar label, .stSidebar section, .stSidebar div, .stRadio label {
                    color: #2b2250 !important;
                }
                .stRadio div[role="radiogroup"] > div {
                    background-color: #fff;
                    color: #2b2250;
                    border-radius: 5px;
                    padding: 6px;
                }
                .stButton>button {
                    background-color: #9370db; color: white; border: none; border-radius: 6px;
                }
                .stButton>button:hover { background-color: #815cc2; }
                h1, h2, h3, h4 { color: #2b2250; }
                .stFileUploader {
                    background-color: #fff !important; border: 1px dashed #9370db !important;
                    border-radius: 8px; color: #444;
                }
                /* Quiz font styling */
                div[data-testid="stRadio"] > label {
                    font-size: 1.2rem !important;  /* Question font size */
                    font-weight: 600 !important;
                    margin-bottom: 0.5rem;
                }
                div[role="radiogroup"] > div > label {
                    font-size: 0.9rem !important;  /* Answer option font size */
                    font-weight: 400 !important;
                }

            </style>
        """, unsafe_allow_html=True)


inject_custom_css(st.session_state.dark_mode)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    prev_mode = st.session_state.dark_mode
    current_input_mode = st.session_state.input_mode
    current_tab = st.session_state.current_tab

    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

    if st.session_state.dark_mode != prev_mode:
        st.session_state.input_mode = current_input_mode
        st.session_state.current_tab = current_tab  # ✅ Preserve current tab
        st.rerun()

    st.markdown("### Navigation")
    st.session_state.current_tab = st.radio("Go to:", ["Home", "Progress", "Saved Notes"], label_visibility="collapsed")
    if st.session_state.current_tab == "Home":
        st.session_state.input_mode = st.selectbox("Study using:", ["Topic", "YouTube Video", "Upload Slides (PDF)"], index=["Topic", "YouTube Video", "Upload Slides (PDF)"].index(st.session_state.input_mode))

# --- MAIN UI ---
st.title("🧠 Agentic Study Assistant")

if st.session_state.current_tab == "Home":
    mode = st.session_state.input_mode

    if mode == "Topic":
        topic = st.text_input("Enter a topic to learn:")
        if st.button("🔍 Generate Study Guide") and topic.strip():
            subtopics = planner_chain.run({"topic": topic})
            query = subtopics.split("\n")[0].split(".", 1)[-1].strip()
            try:
                results = search.run(query)
            except Exception as e:
                st.warning("🔍 DuckDuckGo search failed due to rate limit. Using fallback content.")
                results = f"Overview of {query}: This is a placeholder result used when search fails."

            research = researcher_chain.run({"query": results})
            summary = summarizer_chain.run({"text": research})
            st.session_state.summary = summary
            st.session_state.saved_notes[topic] = summary
            st.markdown("### 📝 Study Notes")
            st.markdown(summary)
            st.session_state.quiz = []  # reset quiz state
            st.session_state.submitted = False

    elif mode == "YouTube Video":
        url = st.text_input("Paste a YouTube lecture URL:")
        if st.button("▶️ Summarize Video"):
            st.success("YouTube summary (stub)")

    elif mode == "Upload Slides (PDF)":
        uploaded_file = st.file_uploader("📄 Upload your lecture slides (PDF only)", type="pdf")
        if uploaded_file and st.button("📘 Summarize PDF"):
            st.success(f"✅ File uploaded: `{uploaded_file.name}`. Summary saved.")
            st.session_state.saved_notes[uploaded_file.name] = f"Summary of PDF: **{uploaded_file.name}**"

    # Show "Generate Quiz" only if summary is available
    if st.session_state.summary and not st.session_state.quiz:
        if st.button("🧪 Generate Quiz"):
            quiz_text = quizzer_chain.run({"text": st.session_state.summary})
            lines = [line.strip() for line in quiz_text.split("\n") if line.strip()]
            questions, answers, explanations = [], [], []
            i = 0
            while i < len(lines):
                if re.match(r"^\d+\.", lines[i]):
                    q = lines[i]
                    opts = lines[i+1:i+5]
                    ans = lines[i+5].replace("Answer:", "").strip()
                    exp = lines[i+6].replace("Explanation:", "").strip()
                    questions.append((q, opts))
                    answers.append(ans)
                    explanations.append(exp)
                    i += 7
                else:
                    i += 1
            st.session_state.quiz = questions
            st.session_state.correct_answers = answers
            st.session_state.explanations = explanations
            st.session_state.user_answers = ["I don't know"] * len(questions)

    # Quiz
    if st.session_state.quiz and not st.session_state.submitted:
        st.markdown("### 🧪 Quiz Time")
        for i, (q, opts) in enumerate(st.session_state.quiz):
            st.session_state.user_answers[i] = st.radio(q, options=opts + ["I don't know"], index=len(opts), key=f"q{i}")
        if st.button("Submit Quiz"):
            score = 0
            for i, ua in enumerate(st.session_state.user_answers):
                if ua.lower().startswith(st.session_state.correct_answers[i][0].lower()):
                    score += 1
            st.session_state.learning_log.append({"topic": topic, "score": score})
            st.session_state.submitted = True
            st.markdown(f"### ✅ You scored **{score}/5**")
            for i, (q, opts) in enumerate(st.session_state.quiz):
                st.markdown(f"**{q}**")
                st.markdown(f"Your answer: `{st.session_state.user_answers[i]}`")
                st.markdown(f"Correct answer: `{st.session_state.correct_answers[i]}`")
                st.markdown(f"📘 Explanation: {st.session_state.explanations[i]}")
            suggestions = suggestion_chain.run({"topic": topic, "score": score})
            st.markdown("#### 📘 Suggested Next Topics")
            for line in suggestions.split("\n"):
                if line.strip():
                    st.markdown(f"- {line.strip()}")
            # Search new topic button after quiz
            if st.session_state.submitted:
                if st.button("🔄 Search New Topic"):
                    for key in ["quiz", "correct_answers", "explanations", "user_answers", "summary", "submitted"]:
                        st.session_state[key] = [] if isinstance(st.session_state[key], list) else ""
                    st.session_state.input_mode = "Topic"
                    st.session_state.current_tab = "Home"
                    st.experimental_rerun()


elif st.session_state.current_tab == "Progress":
    st.subheader("📊 Learning Progress")
    if st.session_state.learning_log:
        for log in st.session_state.learning_log:
            st.markdown(f"- **{log['topic']}**: {log['score']}/5")
    else:
        st.info("No progress logged yet.")

elif st.session_state.current_tab == "Saved Notes":
    st.subheader("📚 Your Saved Notes")
    if st.session_state.saved_notes:
        for title, content in st.session_state.saved_notes.items():
            with st.expander(title):
                st.markdown(content)
    else:
        st.info("No saved notes yet.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by your Agentic Assistant")
