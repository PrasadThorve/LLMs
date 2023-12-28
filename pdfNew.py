import tempfile
import fitz  # PyMuPDF
import chainlit as cl
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

MODEL_PATH = "llama-2-7b-chat.gguf.q4_0.bin"

# Creating a function to load the Llama model
def load_model() -> LlamaCpp:
    """Loads Llama model"""
    # Create a CallbackManager instance with the desired callback(s)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    Llama_model = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True
    )

    return Llama_model

llm = load_model()

template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{question} 
[/INST]
"""

# chainlit code here
@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)

    return llm_chain


@cl.on_message
async def run(message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    if message.file and message.file.type == "application/pdf":
        # Extract text from PDF
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(message.file.content)
            temp_pdf_path = temp_pdf.name
            pdf_text = ""
            with fitz.open(temp_pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pdf_text += page.get_text()
        response = await llm_chain.acall(pdf_text, callbacks=[cb])
    else:
        # Use the input text directly
        response = await llm_chain.acall(message.content, callbacks=[cb])

    if not cb.answer_reached:
        await cl.Message(content=response["text"]).send()
