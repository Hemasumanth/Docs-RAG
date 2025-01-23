from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_single_section(section, chunk_size, chunk_overlap = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)
    
    chunks = text_splitter.create_documents(
        texts=[section["text"]], 
        metadatas=[{"source": section["source"]}])
    
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]


def chunk_sections(sections, chunk_size, chunk_overlap = 100):
    return [
        chunk
        for section in sections
        for chunk in chunk_single_section(section, chunk_size, chunk_overlap)
    ]