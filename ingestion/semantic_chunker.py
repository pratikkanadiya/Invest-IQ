import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from models.model import hf_model, langchain_embeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["SCIPY_USE_LAPACK"] = "0"
os.environ["UV_LINK_MODE"] = "copy"

def read_markdown(markdown_file: str) -> str:
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(markdown_file: str, embeddings) -> list[Document]:
    markdown_content = read_markdown(markdown_file)
    file_path = Path(markdown_file)

    file_metadata = {
        "source_file": file_path.name,
        "file_path": str(file_path),
        "file_extension": file_path.suffix
    }

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    return splitter.create_documents(
        texts=[markdown_content], 
        metadatas=[file_metadata]
    )

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    markdown_file = SCRIPT_DIR.parent / "data" / "markdown" / "2024_Apple.md"
    
    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=langchain_embeddings
    )

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print(f"Metadata Context: {chunk.metadata}") 
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()
