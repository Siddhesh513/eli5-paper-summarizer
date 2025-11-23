#!/usr/bin/env python3
"""
ELI5 Paper Summarizer - CLI Entry Point

Usage:
    python main.py --url https://arxiv.org/abs/2301.00001
    python main.py --url 2301.00001
    python main.py --file paper.pdf
"""
import argparse
import sys
from pathlib import Path

from src.pdf_processor import process_paper, extract_text_from_pdf, detect_sections, PaperContent
from src.chunker import chunk_by_section, prepare_chunks_for_embedding, get_total_tokens
from src.embeddings import create_retriever
from src.summarizer import summarize_paper


def main():
    parser = argparse.ArgumentParser(
        description="ELI5 Paper Summarizer - Transform academic papers into layered summaries"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="arXiv URL or paper ID (e.g., 2301.00001)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to local PDF file",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["all", "technical", "simplified", "eli5"],
        default="all",
        help="Summary level to generate (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )
    
    args = parser.parse_args()
    
    if not args.url and not args.file:
        parser.print_help()
        print("\n❌ Error: Please provide either --url or --file")
        sys.exit(1)
    
    try:
        # Step 1: Process paper
        print("\n" + "="*60)
        print(" 📄 Processing Paper")
        print("="*60)
        
        if args.url:
            print(f"Fetching from arXiv: {args.url}")
            paper = process_paper(args.url)
        else:
            print(f"Reading local file: {args.file}")
            if not Path(args.file).exists():
                print(f"❌ Error: File not found: {args.file}")
                sys.exit(1)
            
            full_text = extract_text_from_pdf(args.file)
            sections = detect_sections(full_text)
            
            paper = PaperContent(
                title=Path(args.file).stem,
                authors=[],
                abstract=sections.get("Abstract", ""),
                full_text=full_text,
                sections=sections,
                pdf_path=args.file,
            )
        
        print(f"✅ Title: {paper.title}")
        print(f"✅ Authors: {', '.join(paper.authors[:3]) if paper.authors else 'Unknown'}")
        print(f"✅ Sections found: {list(paper.sections.keys())}")
        
        # Step 2: Chunk the paper
        print("\n" + "="*60)
        print(" ✂️ Chunking Document")
        print("="*60)
        
        chunks = chunk_by_section(paper.sections)
        texts, metadatas = prepare_chunks_for_embedding(chunks)
        
        total_tokens = get_total_tokens(chunks)
        print(f"✅ Created {len(chunks)} chunks")
        print(f"✅ Total tokens: {total_tokens:,}")
        
        if args.verbose:
            print("\nChunk breakdown:")
            for chunk in chunks:
                print(f"  [{chunk.section}] Part {chunk.chunk_index + 1}/{chunk.total_chunks_in_section}: {chunk.token_count} tokens")
        
        # Step 3: Create embeddings
        print("\n" + "="*60)
        print(" 🔮 Creating Embeddings")
        print("="*60)
        
        retriever = create_retriever(texts, metadatas, paper.title)
        all_chunks = retriever.get_all_chunks()
        print(f"✅ Embedded {len(all_chunks)} chunks in vector store")
        
        # Step 4: Generate summaries
        print("\n" + "="*60)
        print(" 🤖 Generating Summaries")
        print("="*60)
        
        result = summarize_paper(all_chunks)
        
        # Step 5: Display results
        if args.level in ["all", "technical"]:
            print("\n" + "="*60)
            print(" 📚 TECHNICAL SUMMARY")
            print("="*60)
            print(result.technical)
        
        if args.level in ["all", "simplified"]:
            print("\n" + "="*60)
            print(" 📖 SIMPLIFIED SUMMARY")
            print("="*60)
            print(result.simplified)
        
        if args.level in ["all", "eli5"]:
            print("\n" + "="*60)
            print(" 🧒 ELI5 SUMMARY")
            print("="*60)
            print(result.eli5)
        
        # Final stats
        print("\n" + "="*60)
        print(" 📊 Summary Statistics")
        print("="*60)
        print(f"Chunks processed: {result.chunks_used}")
        print(f"Input tokens: {result.token_count:,}")
        print(f"Technical summary: {len(result.technical.split())} words")
        print(f"Simplified summary: {len(result.simplified.split())} words")
        print(f"ELI5 summary: {len(result.eli5.split())} words")
        
        print("\n✅ Done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
