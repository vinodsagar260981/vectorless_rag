"""
PageIndex Vectorless RAG
========================

This script is a cleaned-up, single-file version of the notebook
`01_rag_vectorless.ipynb`.

Core idea:
    Traditional RAG:
        document -> chunks -> embeddings -> vector search -> answer

    PageIndex vectorless RAG:
        document -> semantic tree -> LLM reasons over the tree -> exact sections -> answer

Why this helps:
    Vector similarity is not always the same as relevance. A chunk can share many words with a
    question while still not containing the best answer. This approach asks the LLM to inspect a
    document tree first, then retrieves the selected sections directly.

Required environment variables:
    PAGEINDEX_API_KEY
    GROQ_API_KEY

Example usage:
    python rag_vectorless_single.py --pdf ./Detailed_Summary.pdf --query "What skills are covered?"
    python rag_vectorless_single.py --doc-id existing_doc_id --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pageindex import PageIndexClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TREE_SEARCH_MODEL = "openai/gpt-oss-120b"
DEFAULT_ANSWER_MODEL = "qwen/qwen3.8-27b"
DEFAULT_PDF_PATH = "./Detailed_Summary.pdf"
POLL_SECONDS = 5


def load_api_keys() -> tuple[str, str]:
    """Load API keys from a .env file or the current shell environment."""
    load_dotenv()

    pageindex_api_key = os.getenv("PAGEINDEX_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not pageindex_api_key:
        raise RuntimeError("PAGEINDEX_API_KEY is missing. Add it to your .env file or shell.")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file or shell.")

    return pageindex_api_key, groq_api_key


def create_clients(pageindex_api_key: str, tree_search_model: str) -> tuple[PageIndexClient, ChatGroq]:
    """Create reusable PageIndex and Groq/LangChain clients."""
    pageindex_client = PageIndexClient(api_key=pageindex_api_key)
    tree_search_llm = ChatGroq(model=tree_search_model, temperature=0)
    return pageindex_client, tree_search_llm


# ---------------------------------------------------------------------------
# PageIndex document indexing
# ---------------------------------------------------------------------------

def upload_document(pageindex_client: PageIndexClient, pdf_path: str | Path) -> str:
    """Upload a PDF to PageIndex and return its document id."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    print(f"Uploading: {path}")
    result = pageindex_client.submit_document(str(path))
    doc_id = result["doc_id"]
    print(f"Uploaded. Document ID: {doc_id}")
    return doc_id


def wait_for_tree_index(pageindex_client: PageIndexClient, doc_id: str) -> None:
    """Poll PageIndex until the document tree is ready."""
    print("Building tree index...")
    print("This runs once per document. PageIndex can reuse the cached index later.")

    while True:
        status_result = pageindex_client.get_document(doc_id)
        status = status_result.get("status")
        print(f"Status: {status}")

        if status == "completed":
            print("Tree index ready.")
            return
        if status == "failed":
            raise RuntimeError("PageIndex processing failed. Check that the PDF is valid.")

        time.sleep(POLL_SECONDS)


def fetch_tree(pageindex_client: PageIndexClient, doc_id: str) -> list[dict[str, Any]]:
    """Fetch the complete PageIndex tree with node summaries."""
    tree_result = pageindex_client.get_tree(doc_id, node_summary=True)
    tree = tree_result.get("result", [])
    print(f"Top-level sections: {len(tree)}")
    return tree


# ---------------------------------------------------------------------------
# Tree inspection helpers
# ---------------------------------------------------------------------------

def print_tree(nodes: list[dict[str, Any]], indent: int = 0) -> None:
    """Print the document tree so you can inspect available sections."""
    for node in nodes:
        prefix = "  " * indent + ("- " if indent > 0 else "")
        page = node.get("page_index", "?")
        print(f"{prefix}[{node['node_id']}] {node['title']} (p.{page})")

        child_nodes = node.get("nodes")
        if child_nodes:
            print_tree(child_nodes, indent + 1)


def count_nodes(nodes: list[dict[str, Any]]) -> int:
    """Count all retrievable nodes in the PageIndex tree."""
    total = len(nodes)
    for node in nodes:
        child_nodes = node.get("nodes")
        if child_nodes:
            total += count_nodes(child_nodes)
    return total


def compress_tree_for_prompt(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep only the fields the tree-search LLM needs.

    The full tree can be large. Compression keeps token usage lower while preserving IDs,
    titles, page numbers, and short summaries.
    """
    compressed = []

    for node in nodes:
        entry = {
            "node_id": node["node_id"],
            "title": node["title"],
            "page": node.get("page_index", "?"),
            "summary": node.get("text", "")[:150],
        }

        child_nodes = node.get("nodes")
        if child_nodes:
            entry["children"] = compress_tree_for_prompt(child_nodes)

        compressed.append(entry)

    return compressed


# ---------------------------------------------------------------------------
# LLM-guided tree search
# ---------------------------------------------------------------------------

def parse_llm_json(content: str) -> dict[str, Any]:
    """
    Parse a JSON response from the LLM.

    The prompt asks for JSON only. This helper also handles the common case where a model
    wraps JSON with extra text by extracting the outermost JSON object.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start:end + 1])


def llm_tree_search(query: str, tree: list[dict[str, Any]], llm: ChatGroq) -> dict[str, Any]:
    """
    Ask the LLM to choose the PageIndex node IDs most likely to answer the query.

    Output format:
        {
            "thinking": "...",
            "node_list": ["node_id1", "node_id2"]
        }
    """
    compressed_tree = compress_tree_for_prompt(tree)

    prompt = f"""You are given a query and a document's tree structure, similar to a table of contents.
Your task is to identify which node IDs most likely contain the answer to the query.
Think step by step about which sections are relevant.

Query: {query}

Document Tree:
{json.dumps(compressed_tree, indent=2)}

Reply only in this exact JSON format:
{{
  "thinking": "<your step-by-step reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    response = llm.invoke(prompt)
    return parse_llm_json(response.content) # type: ignore


def find_nodes_by_ids(
    tree: list[dict[str, Any]],
    target_ids: list[str],
) -> list[dict[str, Any]]:
    """Recursively collect tree nodes whose node_id appears in target_ids."""
    found = []

    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)

        child_nodes = node.get("nodes")
        if child_nodes:
            found.extend(find_nodes_by_ids(child_nodes, target_ids))

    return found


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_context(nodes: list[dict[str, Any]]) -> str:
    """Build the grounded context passed to the answer-generation model."""
    context_parts = []

    for node in nodes:
        context_parts.append(
            f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
            f"{node.get('text', 'Content not available.')}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    query: str,
    nodes: list[dict[str, Any]],
    model: str = DEFAULT_ANSWER_MODEL,
) -> str:
    """
    Generate a grounded answer from retrieved PageIndex sections.

    The answer model is instructed to use only the retrieved context and cite the section title
    plus page number for each claim.
    """
    if not nodes:
        return "No relevant sections were found in the document."

    context = build_context(nodes)

    prompt = f"""You are an expert document analyst.
Answer the question using only the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""

    llm = ChatGroq(model=model, temperature=0)
    response = llm.invoke(prompt)
    return response.content # type: ignore


def vectorless_rag(
    query: str,
    tree: list[dict[str, Any]],
    tree_search_llm: ChatGroq,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    verbose: bool = True,
) -> str:
    """
    Run the complete vectorless RAG pipeline.

    Step 1: LLM tree search chooses relevant node IDs.
    Step 2: Node retrieval collects exact sections from the PageIndex tree.
    Step 3: Answer generation writes a grounded, cited response.
    """
    if verbose:
        print("=" * 72)
        print(f"Query: {query}")
        print("=" * 72)

    search_result = llm_tree_search(query, tree, tree_search_llm)
    node_ids = search_result.get("node_list", [])

    if verbose:
        print(f"Reasoning: {search_result.get('thinking', '')[:500]}")
        print(f"Selected node IDs: {node_ids}")

    nodes = find_nodes_by_ids(tree, node_ids)

    if verbose:
        section_titles = [node["title"] for node in nodes]
        print(f"Sections found: {section_titles}")

    answer = generate_answer(query, nodes, model=answer_model)

    if verbose:
        print("\nAnswer:")
        print(answer)

    return answer


# ---------------------------------------------------------------------------
# Command-line workflow
# ---------------------------------------------------------------------------

def interactive_loop(
    tree: list[dict[str, Any]],
    tree_search_llm: ChatGroq,
    answer_model: str,
) -> None:
    """Run repeated document Q&A until the user types quit."""
    print("\nAsk questions about the document. Type 'quit' to exit.\n")

    while True:
        query = input("ASK: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        answer = vectorless_rag(
            query=query,
            tree=tree,
            tree_search_llm=tree_search_llm,
            answer_model=answer_model,
            verbose=False,
        )
        print(f"\nA: {answer}\n")


def parse_args() -> argparse.Namespace:
    """Read command-line options for document indexing and Q&A."""
    parser = argparse.ArgumentParser(description="Run PageIndex vectorless RAG over a PDF.")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to the PDF to upload.")
    parser.add_argument("--doc-id", help="Existing PageIndex document ID to reuse.")
    parser.add_argument("--query", help="Single question to ask the document.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive Q&A loop.")
    parser.add_argument("--show-tree", action="store_true", help="Print the full PageIndex tree.")
    parser.add_argument("--tree-model", default=DEFAULT_TREE_SEARCH_MODEL, help="Groq model for tree search.")
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL, help="Groq model for answer generation.")
    return parser.parse_args()


def main() -> None:
    """Upload or reuse a document, fetch its tree, and answer questions."""
    args = parse_args()

    pageindex_api_key, _groq_api_key = load_api_keys()
    pageindex_client, tree_search_llm = create_clients(pageindex_api_key, args.tree_model)

    if args.doc_id:
        doc_id = args.doc_id
        print(f"Using existing document ID: {doc_id}")
    else:
        doc_id = upload_document(pageindex_client, args.pdf)
        wait_for_tree_index(pageindex_client, doc_id)

    tree = fetch_tree(pageindex_client, doc_id)
    print(f"Total nodes in tree: {count_nodes(tree)}")

    if args.show_tree:
        print("\nDocument Structure:\n")
        print_tree(tree)

    if args.query:
        vectorless_rag(
            query=args.query,
            tree=tree,
            tree_search_llm=tree_search_llm,
            answer_model=args.answer_model,
            verbose=True,
        )

    if args.interactive or not args.query:
        interactive_loop(tree, tree_search_llm, args.answer_model)


if __name__ == "__main__":
    main()
