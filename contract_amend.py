import PyPDF2
import tiktoken
from llama_cpp import Llama
from fpdf import FPDF

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def chunk_text(text, max_tokens=3500):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk))
    
    return chunks

# Load DeepSeek 13B locally
llm = Llama("/Users/sanketjadhav/models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf",n_ctx=8192,verbose=True)

def amend_contract(chunk, user_request):
    prompt = f"""You are a legal AI assistant. Modify the following contract section based on the request.

    Contract Section:
    {chunk}

    User Request:
    {user_request}

    Provide the revised section:
    """
    
    response = llm(prompt=prompt, max_tokens =500)
    return response["choices"][0]["text"].strip()



def save_to_pdf(text, output_path="amended_contract.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Use a Unicode-supported font
    pdf.add_font("DejaVu", "", "/Library/Fonts/Arial Unicode.ttf", uni=True)  # Adjust path if needed
    pdf.set_font("DejaVu", size=12)

    for line in text.split("\n"):
        pdf.cell(200, 10, txt=line.encode("latin-1", "ignore").decode("latin-1"), ln=True)

    pdf.output(output_path, "F")

save_to_pdf("This is an example • with bullet points and special characters — ✅")



pdf_path = "contract.pdf"
user_request = "It says that agreement is under state of California. Make it the state of Miami"
contract_text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(contract_text)

updated_chunks = [amend_contract(chunk, user_request) for chunk in chunks]
final_text = "\n\n".join(updated_chunks)

save_to_pdf(final_text, "amended_contract.pdf")
print("Amended contract saved as amended_contract.pdf")
