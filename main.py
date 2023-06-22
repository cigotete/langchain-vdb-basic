import os
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import dotenv_values  
dotenv_values()

cwd = os.getcwd()
path_rel_text = "text_source/vector_db.txt"
path_abs_text = os.path.join(cwd, path_rel_text)

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(
    api_key=pinecone_api_key,
    environment='us-west1-gcp-free'
    )


if __name__ == '__main__':
    print('Hello Vector DB!')
    loader = TextLoader(path_abs_text)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # created an index with 1536 dimensions, value taken from text-embedding-ada-002
    # https://platform.openai.com/docs/guides/embeddings/what-are-embeddings,
    # and euclidean as metric for measure similarity.
    docsearch = Pinecone.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name='medium-blogs-embeddings-index'
        )
    
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=docsearch,
        return_source_documents=True
      )
    query = "Vector database is an innovative way to communicate with who?."
    result = qa({"query": query})
    print(result)