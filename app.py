# app.py
import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

INDEX_DIR = "storage/faiss_index"

SYSTEM_PROMPT = """You are an AI Teaching Assistant for a Master-level course in data visualization using Tableau.
Answer ONLY using the provided context and any information about Tableau Desktop that you can find. You will come across these situations
1. If the students ask you about sample certificate questions use all the information on Tableau to generate sample multiple-choice
multiple-answer questions. In general, if they say certificate exam or just exam or salesforce exam,
they mean the Salesforce Certified Tableau Desktop Foundations and you should be using the exam details file, the study guide and tableau information
first to answer their questions. If they ask you about exam sample questions give them a mix of multiple choice (choose 1 out of 4-5 options) and multiple select questions. For 
the multiple select questions give them 5 options and they must select 2,3 or 4 of them to be correct. You must tell them how many they need to select.
In those questions try to include multiple ways of doing the same task in Tableau. Don't show them the correct answers. After generating the questions ask them if they want the 
correct answers. If they say "yes" then give them the answers.
2. The students are asking you about help with the class. Do you best to answer using the provided context only. Be polite and if they don't understand suggest they ask Detelina
3. If you are unsure how to answer in the given context apologize and tell them to talk to Detelina.
Keep answers concise and student-friendly. 
"""

# Updated: include conversation history with MessagesPlaceholder
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nReturn a clear answer.")
])

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def format_context(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"{src} p.{page+1}" if isinstance(page, int) else src
        snippet = d.page_content.strip().replace("\n", " ")
        lines.append(f"[{i}] ({tag}) {snippet[:800]}")
    return "\n\n".join(lines)

def format_citation(meta):
    src = meta.get("source", "unknown")
    page = meta.get("page", None)
    return f"{src} p.{page+1}" if isinstance(page, int) else src

def main():
    st.set_page_config(page_title="BSAN 720 Tutoring Center", page_icon="🎓")
    st.markdown('<p style="font-size:30px; color:#646464;">🎓 BSAN 720 Tutoring Center</p>', unsafe_allow_html=True)

    if "OPENAI_API_KEY" not in st.secrets and not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()

    vs = load_vectorstore()
    retriever_k = 5
    threshold = 0.5

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # if "history" not in st.session_state:
    #     st.session_state.history = []
    #     if "history" not in st.session_state:
    #         # Add greeting + fun fact when session starts
    #         # Ask the LLM for a fun fact
    #         fact_prompt = "Give me an interesting piece of information from the Tableau resources. Start with 'Did you know' and keep it short"
    #         fun_fact = llm.invoke(fact_prompt).content
    #
    #         with st.chat_message("assistant"):
    #             st.markdown(
    #                  f"Hi! I'm here to help you with BSAN 720.\n\n"
    #                 # f"Ask me anything about Tableau, the course, or the certificate exam.\n\n"
    #                  f"💡 **Tableau fact:** {fun_fact}"
    #             )

    # Make sure history exists
    if "history" not in st.session_state:
        st.session_state.history = []

    # Show greeting + fun fact only once per session (keep separate from history)
    if "greeted" not in st.session_state:
        fact_prompt = "Tell me a Tableau Desktop fact that might show up on the certificate exam. Start with 'Did you know' and keep it short"
        fun_fact = llm.invoke(fact_prompt).content
        with st.chat_message("assistant", avatar="icons/Maya_icon.png"):
            st.markdown(
                f"Hi! I'm Maya. I'm here to help you with BSAN 720.\n\n"
                f"💡 **Tableau fact:** {fun_fact}"
            )
        st.session_state.greeted = True  # flag so it doesn’t repeat

    # Replay chat history in the UI
    for role, msg in st.session_state.history:
        if role == 'user':
            with st.chat_message(role, avatar='🤓'):
                st.markdown(msg)
        else:
            with st.chat_message(role, avatar='icons/Maya_icon.png'):
                st.markdown(msg)

    #user_q = st.chat_input("Ask about the course material, syllabus, Tableau, certificate exam ... ")
    user_q = st.chat_input("...")
    if not user_q:
        st.info("Tip: ask “What is Detelina's email” or “Give me 5 sample questions for the certificate exam.”")
        return

    with st.chat_message("user", avatar='🤓'):
        st.markdown(user_q)
    st.session_state.history.append(("user", user_q))

    with st.spinner("Thinking…"):
        # Retrieve documents
        docs_scores = vs.similarity_search_with_score(user_q, k=retriever_k)
        filtered = [(d, s) for d, s in docs_scores if s <= (1 - threshold)]
        docs = [d for d, _ in (filtered or docs_scores)]

        if not docs:
            bot = "I couldn’t find this in the course materials. Please check the syllabus or slides index (or ask Detelina)."
        else:
            context = format_context(docs)

            # Build chat_history as LangChain messages
            chat_history = []
            for role, msg in st.session_state.history[:-1]:  # exclude current input
                if role == "user":
                    chat_history.append(HumanMessage(content=msg))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=msg))

            prompt = ANSWER_PROMPT.format_messages(
                question=user_q,
                context=context,
                chat_history=chat_history
            )
            resp = llm.invoke(prompt)
            bot = resp.content

    with st.chat_message("assistant", avatar="icons/Maya_icon.png"):
        st.markdown(bot)
    st.session_state.history.append(("assistant", bot))

if __name__ == "__main__":
    main()
