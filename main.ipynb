!pip install -U langchain colpali-engine transformers faiss-cpu gradio pdf2image pillow torch
!pip install git+https://github.com/illuin-tech/colpali.git
!pip install -U langchain-community langchain-huggingface
!apt-get update && apt-get install -y poppler-utils


from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
import torch
from pdf2image import convert_from_path
import tempfile, os
from langchain.embeddings.base import Embeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

class NomicMultimodalEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-multimodal-3b"):
        self.model = BiQwen2_5.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
        self.processor = BiQwen2_5_Processor.from_pretrained(model_name)

    def embed_documents(self, texts):
      batch = self.processor.process_queries(texts).to(self.model.device)
      with torch.no_grad():
          # Convert from bfloat16 to float32 before numpy conversion
          emb = self.model(**batch).float()
      return emb.cpu().numpy().tolist()


    def embed_query(self, text):
        return self.embed_documents([text])[0]

def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, dpi=150)

def image_to_text_chunks(images):
    return [f"[PAGE_IMAGE]{i}" for i in range(len(images))]

images = pdf_to_images("annual_report.pdf")    #Change the file path to the actual one.
document_chunks = image_to_text_chunks(images)


embedder = NomicMultimodalEmbeddings()
vector_store = FAISS.from_texts(document_chunks, embedder)

# Load Qwen 0.6B
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)
result = qa_chain("What is the main information present in the report?")
print(result["result"])
