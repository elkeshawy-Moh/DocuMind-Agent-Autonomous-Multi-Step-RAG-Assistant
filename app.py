# ==============================================================================
# app.py: GRADIO PROFESSIONAL USER INTERFACE & APPLICATION ENTRY POINT
# ==============================================================================

import os
import gradio as gr
from ingest import process_uploaded_pdfs
from agent import run_documind_agent

custom_css = """
.header-banner {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 15px;
}
.header-banner h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
}
.header-banner p {
    margin: 5px 0 0 0;
    font-size: 15px;
    opacity: 0.9;
}
"""

def set_api_key(api_key: str) -> str:
    """Allows users to set their Groq API Key directly from the UI if not in .env"""
    if not api_key.strip():
        return "⚠️ Please enter a valid Groq API Key."
    os.environ["GROQ_API_KEY"] = api_key.strip()
    return "✅ Groq API Key configured successfully!"

def handle_chat_interaction(user_message, chat_history):
    """Handles user message submission, runs the agent, and updates UI components."""
    if not user_message.strip():
        return "", chat_history, ""

    answer, trace = run_documind_agent(user_message)
    chat_history.append((user_message, answer))
    return "", chat_history, trace

# Build the Gradio Blocks Interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=custom_css,
    title="DocuMind Agent"
) as demo:

    gr.HTML("""
    <div class="header-banner">
        <h1>🧠 DocuMind Agent: Autonomous Multi-Step RAG Assistant</h1>
        <p>Intelligent Query Planning • Dynamic Tool Routing • Multi-Document Comparison • Verified Citations</p>
    </div>
    """)

    with gr.Tabs():
        # ======================================================================
        # TAB 1: KNOWLEDGE BASE SETUP (INGESTION PIPELINE)
        # ======================================================================
        with gr.TabItem("📁 1. Knowledge Base Setup"):
            gr.Markdown("### Step A: Configure API Key (Optional if set in environment)")
            with gr.Row():
                api_input = gr.Textbox(
                    placeholder="Enter your GROQ_API_KEY (gsk_...)",
                    type="password",
                    show_label=False,
                    scale=4
                )
                api_btn = gr.Button("🔑 Save API Key", scale=1)
            api_status = gr.Markdown("")
            api_btn.click(fn=set_api_key, inputs=[api_input], outputs=[api_status])

            gr.Divider()
            gr.Markdown("### Step B: Upload Documents & Build Vector Index")
            gr.Markdown("Upload one or multiple PDF files. The pipeline will extract text, clean noise, create semantic chunks, and index them into ChromaDB.")

            with gr.Row():
                with gr.Column(scale=2):
                    pdf_uploader = gr.File(
                        label="Upload PDF Documents",
                        file_count="multiple",
                        file_types=[".pdf"]
                    )
                    build_btn = gr.Button("⚙️ Build & Index Knowledge Base", variant="primary", size="lg")

                with gr.Column(scale=1):
                    status_box = gr.Markdown(
                        value="⏳ **Status:** Waiting for PDF documents to be uploaded...",
                        label="Pipeline Status"
                    )

            build_btn.click(
                fn=process_uploaded_pdfs,
                inputs=[pdf_uploader],
                outputs=[status_box]
            )

        # ======================================================================
        # TAB 2: AGENTIC WORKSPACE (CHAT + REASONING TRACE)
        # ======================================================================
        with gr.TabItem("🤖 2. Agentic Workspace"):
            with gr.Row():
                # Left Column: Interactive Conversational Interface
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="DocuMind Conversation",
                        height=480,
                        show_copy_button=True
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a complex multi-step question (e.g., Compare metrics across documents, calculate growth, summarize findings)...",
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("🚀 Ask Agent", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ Clear Conversation", size="sm")

                # Right Column: Live Transparency Panel
                with gr.Column(scale=2):
                    gr.Markdown("### 🔍 Agent Reasoning & Tool Execution Trace")
                    gr.Markdown("*Watch how the agent plans, refines queries, and selects tools in real time:*")
                    reasoning_panel = gr.Markdown(
                        value="Waiting for your question to display step-by-step agent reasoning...",
                        elem_id="reasoning-trace"
                    )

            gr.Examples(
                examples=[
                    "Summarize the main objectives and key findings across all uploaded documents with citations.",
                    "Compare the key figures mentioned in the documents and calculate the percentage difference or growth rate.",
                    "What are the primary challenges mentioned in the files, and what solutions are proposed?"
                ],
                inputs=msg_input,
                label="💡 Click an Example Multi-Step Question to Test:"
            )

            send_btn.click(
                fn=handle_chat_interaction,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, reasoning_panel]
            )
            msg_input.submit(
                fn=handle_chat_interaction,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, reasoning_panel]
            )
            clear_btn.click(
                fn=lambda: ([], "Conversation cleared. Ready for new questions."),
                outputs=[chatbot, reasoning_panel]
            )

if __name__ == "__main__":
    demo.launch(share=True)
