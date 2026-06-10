"""Quick smoke test: parse a real book and show stats."""

from chatbot.ingestion.parser import parse_pdf

book = "./Books/Municipal Solid Waste Management in Developing Countries ( PDFDrive ).pdf"

print("Parsing real book...")
doc = parse_pdf(book)

print(f"Title: {doc.title}")
print(f"Total pages: {doc.total_pages}")
print(f"Extracted pages (with text): {len(doc.pages)}")

if len(doc.pages) > 5:
    print(f"\nSample page 6:")
    print(f"  Chapter: {doc.pages[5].chapter}")
    print(f"  Page number: {doc.pages[5].page_number}")
    print(f"  Text preview: {doc.pages[5].text[:300]}...")

# Now chunk it
from chatbot.ingestion.chunker import chunk_document

chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=200, min_chunk_size=100)
print(f"\nChunks created: {len(chunks)}")
if chunks:
    print(f"First chunk preview:")
    print(f"  Book: {chunks[0].book_title}")
    print(f"  Chapter: {chunks[0].chapter}")
    print(f"  Page range: {chunks[0].page_range}")
    print(f"  Text: {chunks[0].text[:200]}...")

print("\nSmoke test PASSED!")
