# ui_utils.py

import streamlit as st

def render_header():
    st.set_page_config(page_title="Agentic Study Assistant", layout="centered")
    st.title("🧠 Agentic Study Assistant")

def render_subtopics(subtopics: str):
    st.subheader("🧭 Subtopics Breakdown")
    st.markdown(subtopics)

def render_summary(summary: str):
    st.subheader("📝 Study Notes")
    st.markdown(summary)

def render_suggestions(suggestions: str):
    st.subheader("📘 Suggested Topics to Explore")
    for line in suggestions.split("\n"):
        if line.strip():
            st.markdown(f"- {line.strip()}")

def render_quiz_results(questions, user_answers, correct_answers, explanations):
    st.subheader("✅ Results")
    score = 0
    for i, (question, _) in enumerate(questions):
        user_ans = user_answers[i]
        correct_ans = correct_answers[i]
        explanation = explanations[i]
        is_correct = user_ans.lower().startswith(correct_ans[0].lower())
        if is_correct:
            score += 1
        st.markdown(f"**{question}**")
        st.markdown(f"Your answer: `{user_ans}`")
        st.markdown(f"Correct answer: `{correct_ans}`")
        st.markdown(":green[Correct]" if is_correct else ":red[Incorrect]")
        st.markdown(f"📘 Explanation: {explanation}")
        st.markdown("---")
    st.markdown(f"### 🧮 Your Score: **{score}/5**")
    return score

def render_footer():
    if st.button("🔄 Look up a new topic"):
        for key in [
            "quiz_questions", "correct_answers", "explanations", "summary",
            "user_answers", "submitted"
        ]:
            st.session_state[key] = [] if isinstance(st.session_state.get(key), list) else ""
        st.session_state.submitted = False
        st.experimental_rerun()
