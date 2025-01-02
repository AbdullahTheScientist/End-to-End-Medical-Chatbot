from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from src.helper import download_hugging_face_embeddings, text_split, load_pdf_file
from langchain.vectorstores import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
)