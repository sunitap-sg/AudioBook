import pdfplumber
from dotenv import load_dotenv
from langchain import ChatOllama
from openai import OpenAI
import os
import json
import re

load_dotenv()

client = OpenAI()
def extract_text(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    # return full_text
    print(f"the first few text..{full_text[:5000]}")
    return full_text[0:3000]

def extract_contents_Section(full_text):
    pattern = r"(CONTENTS|table of contents)"
    match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
    print(f"the match.....{match}")
    if not match:
        return None
        # print(f"Contents section found: {match.group(1)}, and {match.group(2)}")
        # return match.group(2).strip()

    start = match.end()

    contents = full_text[start:start+3000]
    print(f"Contents section found: {contents}")
    return contents


def get_chapter_from_content(content):
    #without lllm
    pass



def safe_json_parse(content):
    if not content:
        raise ValueError("OpenAI returned empty response")

    # Remove ```json ... ``` wrappers
    content = re.sub(r"```json", "", content)
    content = re.sub(r"```", "", content)
    content = content.strip()

    return json.loads(content)

def get_chapter_markers(book_text):
    prompt = f"""
    The following text is from a book.

    Identify all chapters based on the content, do not analyze the whole file if it has content page. For each chapter, return the title and the exact first sentence of the chapter.
    
    Return JSON in this format:

    [
      {{
        "title": "Chapter Title",
        "first_line": "Exact first sentence of chapter"
      }}
    ]

    Only return valid JSON.

    Book text:
    {book_text[:25000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content
    print(f"the raw content.....{content}")

    # content = content.strip()
    # if content.startswith("```"):
    #     content = content.split("```")[1]

    # return json.loads(content)
    return safe_json_parse(content)


def split_book_locally(full_text, markers):
    chapters = []

    for i, marker in enumerate(markers):
        title = marker["title"]
        first_line = marker["first_line"]

        start_index = full_text.find(first_line)

        if start_index == -1:
            continue

        if i + 1 < len(markers):
            next_first_line = markers[i + 1]["first_line"]
            end_index = full_text.find(next_first_line)
        else:
            end_index = len(full_text)

        chapter_content = full_text[start_index:end_index].strip()

        chapters.append({
            "title": title,
            "content": chapter_content
        })

    return chapters

import re

def safe_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:100]

def save_chapters(book_name, chapters):
    base_folder = "parsed_books"
    book_folder = os.path.join(base_folder, book_name)

    os.makedirs(book_folder, exist_ok=True)

    for chapter in chapters:
        title = safe_filename(chapter["title"])
        file_path = os.path.join(book_folder, f"{title}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chapter["content"])

    print(f"Chapters saved in {book_folder}")

def normalize_text(text):
    text = text.replace("```")

def process_book(pdf_path):
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print("Extracting text...")
    text = extract_text(pdf_path)

    print("Splitting chapters using AI...")
    chapters = get_chapter_markers(text)
    markers = get_chapter_markers(text)
    chapters = split_book_locally(text, markers)


    print("Saving chapters...")
    save_chapters(book_name, chapters)
# 
    print("Done!")


if __name__ == "__main__":
    pdf_path = r"C:\Users\baps\sunita\Test\input_pdf\The Little Book of Good Thi_ (Z-Library).pdf"
    extract_text(pdf_path)
    # extract_contents_Section(pdf_path)
    process_book(pdf_path)
