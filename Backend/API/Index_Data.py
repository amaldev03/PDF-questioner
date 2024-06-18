from io import BytesIO
from fastapi import  File, UploadFile
import PyPDF2
from langchain.text_splitter import LatexTextSplitter
import torch
from Config import allcreds
   

# Get Creds from config.py

es_client = allcreds["es"]
groq_client = allcreds["client"]
colbert_creds=allcreds["colbert"]
tokenizer_creds=allcreds["tokenizer"]


# Class to extract file content and index in elastic.

class Extract_text:

    def __init__(self):
        pass

    
    def IndexData(self,filename,docstring,count,embeddings):

        jsondata ={                                                                                                 # Index data into elastic search
            "filename":filename,
            "docstring":docstring,
            "count":count,
            "embeddings":embeddings
        }

        es_client.index(index="chatbot_index", document=jsondata)


    def vector_creation(self,text_chunks,filename):

        count=0
        tokens = tokenizer_creds(text_chunks, padding=True, truncation=True, return_tensors="pt")                   # Tokenize text chunks
        with torch.no_grad():                                                                                       # Get embeddings for the passed chunks using colbert model
            embeddings = colbert_creds(**tokens).last_hidden_state.mean(dim=1).numpy()[0]                           # Compute embeddings and convert to NumPy array 
            self.IndexData(filename,text_chunks,count,embeddings)                                                   # Store the embeddings with associated data
            count+=1
        
        # Convert bytes to a readable blob for PyPDF2
    async def Extract_text(self,file: UploadFile = File(...)):
        try:
            contents =await file.read()                                                                             # Read file content.
            file_name=file.filename      
            print(file_name)                                                                           # Get file content
        except Exception:
            return {"message": "There was an error reading the file"}

        blob_to_read = BytesIO(contents)                                                                            # convert content into blob. 
        file_reader = PyPDF2.PdfReader(blob_to_read)                                                                # Read the blob using PdfReader
        text_content = ""

        for page_num in range(len(file_reader.pages)):                                                              # Loop through each pages and extract text.
            page = file_reader.pages[page_num]
            text_content += page.extract_text()
        
        # Load a pre-trained sentence transformer model
        latex_splitter = LatexTextSplitter(chunk_size=2000, chunk_overlap=0)                                        # Split the content into chunks
        docs = latex_splitter.create_documents(texts=[text_content])                                                # Create a list of chunks
        
        for doc in docs:                                                      
            docstring=doc.page_content                                         
            self.vector_creation(docstring,file_name)                                                               # Passing each chunk for vector creation


Index_data=Extract_text()                                                                                      # Create instance for Extract_text_Index class
   