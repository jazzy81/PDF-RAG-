from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "./data/BOIR_Filing_Instructions.pdf"
model = "llama3.2"

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading...")
else:
    print("upload a PDF file")

content = data[0].page_content
# print(content[:1000])

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("done splitting....")

print(f"Number of chunks : {len(chunks)}")
print(f"Example chunk: {chunks[0]}")

# import ollama 
# ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)

print("done adding to vector databases...")

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model=model)

from langchain.prompts import PromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome 
some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# res = chain.invoke(input=("what is the document about?",))
# res = chain.invoke(
#     input=("what are the main points as a business owner I should be aware of?",)
# )
res = chain.invoke(input=("how to report BOI?",))

print(res)