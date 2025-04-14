import pdfplumber
import tiktoken
from llama_cpp import Llama
from docx import Document
from dotenv import load_dotenv
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text



def chunk_text(text, max_tokens=3500):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk))
    
    return chunks

# Load Llama 8B locally
load_dotenv()
llm_path = os.getenv("LLAMA_MODEL_PATH")
llm = Llama(model_path=llm_path, n_ctx=4096)  # or higher if supported

def amend_contract(chunk, user_request):
    prompt = f"""You are a legal contract editor AI. Carefully follow the instruction below to modify the contract clauses **only where required**.

--- CONTRACT SECTION START ---
{chunk}
--- CONTRACT SECTION END ---

--- USER REQUEST ---
{user_request}
--- END REQUEST ---

Please revise ONLY the relevant parts and return the updated section. Do not remove or shorten any unrelated content.
"""
    response = llm(prompt=prompt, max_tokens=1500)  # Increase tokens for longer context
    return response["choices"][0]["text"].strip()


def save_to_docx(text, output_path="amended_contract.docx"):
    doc = Document()
    for para in text.split("\n\n"):
        doc.add_paragraph(para.strip())
    doc.save(output_path)


pdf_path = "contract.pdf"
user_request = "In some parts, it says that agreement is under state of California. Only in those parts, please change it to the state of Miami"
contract_text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(contract_text)

updated_chunks = [amend_contract(chunk, user_request) for chunk in chunks]
final_text = "\n\n".join(updated_chunks)

save_to_docx(final_text, "amended_contract.docx")
print("Amended contract saved as amended_contract.docx")
