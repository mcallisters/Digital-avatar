import os
import re
import json

INPUT_DIR = "data/extracted_texts"
OUTPUT_FILE = "data/json_outputs/all_publications.json"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def parse_publication(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract title (assume first non-empty line)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    title = lines[0] if lines else ""

    # Extract authors (line(s) after title until Introduction)
    author_lines = []
    for line in lines[1:10]:  # look at next 10 lines after title
        if re.search(r"Introduction", line, re.IGNORECASE):
            break
        author_lines.append(line)
    authors = " ".join(author_lines).strip()

    # Extract sections by headings
    section_headers = ["Introduction", "Materials and methods", "Results", "Discussion", "Conclusion", "References", "Keywords", "DOI"]
    sections = {}
    for i, header in enumerate(section_headers):
        pattern = rf"{header}\s*(.*?)\s*(?=\n({'|'.join(section_headers)})\s|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[header] = match.group(1).strip()

    return {
        "title": title,
        "authors": authors,
        "sections": sections
    }

def main():
    all_publications = []

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(INPUT_DIR, filename)
            pub_data = parse_publication(file_path)
            all_publications.append(pub_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as jf:
        json.dump(all_publications, jf, ensure_ascii=False, indent=4)

    print(f"Saved JSON for {len(all_publications)} publications to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
