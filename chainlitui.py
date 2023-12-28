

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
import chainlit as cl

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
Give me short and precise answers.
{question} 
[/INST]
"""

# chainlit code here
@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm,verbose=True)
    cl.user_session.set("llm_chain", llm_chain)

    return llm_chain


@cl.on_message
async def run(input_str):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    res = await llm_chain.acall(input_str, callbacks=[cb])

    if not cb.answer_reached:
        await cl.Message(content=res["text"]).send()