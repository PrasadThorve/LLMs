from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.llms import LlamaCpp
from langchain.llms import CTransformers

MODEL_PATH = "llama-2-7b-chat.ggmlv3.q2_K.bin"

# Creating a function to load the Llama model
def load_model() -> CTransformers:
    """Loads Llama model"""
    # Create a CallbackManager instance with the desired callback(s)
    callback_manager: CallbackManager([StreamingStdOutCallbackHandler()])
  

    Llama_model = CTransformers(
        model=MODEL_PATH,
        temperature=0.5,

        max_tokens=2000,
        top_p=1,
        
    )

    return Llama_model

llm = load_model()

model_prompt = """
Question: which coutry has the largest population?
"""

response = llm.generate(prompt=model_prompt)
 