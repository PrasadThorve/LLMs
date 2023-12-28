from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

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

model_prompt = """
Question: how can you help me?
"""

response = llm(model_prompt)
