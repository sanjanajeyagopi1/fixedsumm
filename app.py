import PyPDF2
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI environment variables from .env file
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_type = os.getenv("OPENAI_API_TYPE")
azure_api_version = os.getenv("OPENAI_API_VERSION")

# Set the environment variables in the current environment (if needed)
os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
os.environ["OPENAI_API_TYPE"] = azure_api_type
os.environ["OPENAI_API_VERSION"] = azure_api_version

def extract_pdf_metadata(pdf_path):
    pdf_document = PyPDF2.PdfReader(pdf_path)
    metadata = pdf_document.metadata
    first_page_text = pdf_document.pages[0].extract_text()
    return metadata, first_page_text

def chunk_text(text, chunk_size=4000):
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunks = []
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_prompt_for_chunk(metadata, chunk_text, chunk_num):
    chunked_prompt_template = """
    Chunk {chunk_num}:
    
    Please extract the following metadata and generate a summary for this chunk of the document:
    Title: {title}
    Authors: {authors}
    Keywords: {keywords}
    
    Here is the content of this chunk:
    {text}
    
    Provide a detailed summary of 30-35 lines that helps a reader quickly understand the core ideas, objectives, methodology, and key findings of the document. The summary should provide enough detail for someone to grasp the overall content in a 2-minute read.
    """
    prompt = PromptTemplate(
        input_variables=["chunk_num", "title", "authors", "keywords", "text"],
        template=chunked_prompt_template
    )
    
    title = metadata.get('title', 'Title not found')
    authors = metadata.get('author', 'Authors not found')
    keywords = metadata.get('keywords', 'Keywords not found')
    
    return prompt.format(chunk_num=chunk_num, title=title, authors=authors, keywords=keywords, text=chunk_text)

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    model_name="gpt-4o-mini"
)

def generate_summary(prompt):
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()

def limit_summary_to_lines(summary, max_lines=35):
    lines = summary.split("\n")
    return "\n".join(lines[:max_lines])

def pdf_summarizer_with_chunking(pdf_path, chunk_size=4000, max_lines_per_summary=35):
    metadata, first_page_text = extract_pdf_metadata(pdf_path)
    chunks = chunk_text(first_page_text, chunk_size=chunk_size)
    
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        prompt = create_prompt_for_chunk(metadata, chunk, i+1)
        summary = generate_summary(prompt)
        limited_summary = limit_summary_to_lines(summary, max_lines=max_lines_per_summary)
        chunk_summaries.append(limited_summary)
    
    final_summary = "\n\n".join(chunk_summaries)
    return final_summary

def main():
    st.title("PDF Summarizer with Chunking")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        pdf_path = uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Processing PDF...")
        summary = pdf_summarizer_with_chunking(pdf_path, chunk_size=4000, max_lines_per_summary=35)
        st.write("### Summary:")
        st.text_area("Summary", summary, height=600)

if __name__ == "__main__":
    main()
