from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

url = 'http://localhost:6333'
collection_name = "llm_vdb"

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)

DATA_PATH = '../data/'

#create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 10)
    texts = text_splitter.split_documents(documents)
    print(f"texts are {texts}")
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device':'cpu'} )
    

    qdrant = Qdrant.from_documents(documents,
                                    embeddings,
                                    url= url,
                                    prefer_grpc= False,
                                    force_recreate=True,
                                    collection_name=collection_name)


if __name__ == '__main__':
    create_vector_db()