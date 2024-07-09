from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# TODO: ConversationalRetrievalChain is deprecated - update using `create_retrieval_chain`

def get_llm(client):
    model_kwargs = {
        "max_tokens": 512,
        "temperature": 0, 
        "top_k": 250, 
        "top_p": 1, 
        "stop_sequences": ["\n\nHuman:"] 
    }
    
    llm = BedrockChat(
        client = client,
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs = model_kwargs
    )
    
    return llm

def get_index(client): 
    embeddings = BedrockEmbeddings(client = client)
    pdf_path = "2022-Shareholder-Letter.pdf"
    loader = PyPDFLoader(file_path = pdf_path)

    # split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
    # divide into 1000-character chunks using the separators
    # overlap 100 characters with previous chunk
    text_splitter = RecursiveCharacterTextSplitter( 
        separators = ["\n\n", "\n", ".", " "], 
        chunk_size = 1000, 
        chunk_overlap = 100
    )

    # use an in-memory vector store for demo purposes
    index_creator = VectorstoreIndexCreator( 
        vectorstore_cls = FAISS, 
        embedding = embeddings, 
        text_splitter = text_splitter,
    )
    
    index_from_loader = index_creator.from_loaders([loader])
    
    return index_from_loader

def get_memory():
    """
    create memory for this chat session
    """

    # Maintains a history of previous messages
    memory = ConversationBufferWindowMemory(memory_key = "chat_history", return_messages = True)
    return memory


def get_rag_chat_response(client, input_text, memory, index):
    """
    chat client function
    """
    
    llm = get_llm(client)
    
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory, verbose=True,
                                                                        return_source_documents=True)

    # pass the user message and summary to the model
    chat_response = conversation_with_retrieval.invoke({"question": input_text})

    print(chat_response)
    
    return chat_response['answer']


