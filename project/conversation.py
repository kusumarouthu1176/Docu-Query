from langchain.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def setup_conversation_chain(vector_store):
    """Sets up a conversational chain using Google PALM."""
    llm = GooglePalm()
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain(llm=llm, retriever=retriever, memory=memory)
