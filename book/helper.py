import logging
import os
import json
import re

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_PARSED_FOLDER: str = os.getenv("PARSED_BOOKS_FOLDER", "parsed_books")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
load_dotenv()
client = OpenAI()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Remove markdown code-fence markers and strip surrounding whitespace.

    Args:
        text: Raw string that may contain triple-backtick fences.

    Returns:
        Cleaned string with fences removed.
    """
    text = text.replace("```", "")
    return text.strip()


def safe_filename(name: str) -> str:
    """Sanitize a string so it can be used safely as a file name.

    Removes characters that are illegal on common file-systems and replaces
    spaces with underscores.  Truncates the result to 100 characters to avoid
    overly long paths.

    Args:
        name: The proposed file/directory name.

    Returns:
        A sanitized file name string no longer than 100 characters.

    Raises:
        ValueError: If *name* is empty after sanitization.
    """
    if not isinstance(name, str) or not name.strip():
        logger.warning("safe_filename received an empty or invalid input: %r", name)
        raise ValueError("File name must be a non-empty string.")

    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized[:100]

    if not sanitized:
        logger.warning("File name became empty after sanitization; original: %r", name)
        raise ValueError("File name is empty after sanitization.")

    return sanitized


def safe_json_parse(content: str) -> list:
    """Parse a JSON string that may be wrapped in markdown code fences.

    Args:
        content: Raw string returned by the LLM, potentially wrapped in
                 triple backticks.

    Returns:
        Parsed Python object (expected to be a list of chapter dicts).

    Raises:
        ValueError: If *content* is empty.
        json.JSONDecodeError: If the content is not valid JSON after cleaning.
    """
    if not content or not content.strip():
        raise ValueError("LLM returned an empty response.")

    # Strip markdown code-fence wrappers
    content = re.sub(r"```json", "", content)
    content = re.sub(r"```", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON from LLM response: %s", exc)
        raise


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        The first 3 000 characters of the concatenated page text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: Re-raises any pdfplumber errors.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception:
        logger.exception("Error reading PDF: %s", pdf_path)
        raise

    logger.debug("First 5 000 chars of extracted text:\n%s", full_text[:5000])
    return full_text[:3000]


def extract_contents_section(full_text: str) -> str | None:
    """Locate and return the table-of-contents block within *full_text*.

    Args:
        full_text: Raw text extracted from the PDF.

    Returns:
        Up to 3 000 characters of text starting after the 'CONTENTS' heading,
        or ``None`` if no contents section is found.
    """
    pattern = r"(CONTENTS|table of contents)"
    match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
    logger.debug("Contents section match: %s", match)

    if not match:
        return None

    start = match.end()
    contents = full_text[start: start + 3000]
    logger.debug("Contents section found:\n%s", contents)
    return contents


# ---------------------------------------------------------------------------
# Chapter detection (AI-assisted)
# ---------------------------------------------------------------------------

def get_chapter_markers(book_text: str) -> list:
    """Use the OpenAI API to identify chapter titles and first sentences.

    Args:
        book_text: Extracted book text (first ~25 000 chars are sent).

    Returns:
        A list of dicts with keys ``"title"`` and ``"first_line"``.

    Raises:
        ValueError: If the API returns an empty response.
        json.JSONDecodeError: If the response is not valid JSON.
    """
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
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_content = response.choices[0].message.content
    logger.debug("Raw LLM response:\n%s", raw_content)
    return safe_json_parse(raw_content)


def get_chapter_from_content(content: str) -> None:
    """Placeholder for a future rule-based chapter extractor (no LLM).

    Args:
        content: Table-of-contents text extracted from the PDF.
    """
    # TODO: implement regex/heuristic chapter detection without an LLM
    pass


# ---------------------------------------------------------------------------
# Chapter splitting and saving
# ---------------------------------------------------------------------------

def split_book_locally(full_text: str, markers: list) -> list:
    """Split *full_text* into chapters using the provided chapter markers.

    Args:
        full_text: The complete book text.
        markers: List of dicts with ``"title"`` and ``"first_line"`` keys,
                 as returned by :func:`get_chapter_markers`.

    Returns:
        List of dicts, each containing ``"title"`` and ``"content"`` keys.
    """
    chapters = []

    for i, marker in enumerate(markers):
        title = marker["title"]
        first_line = marker["first_line"]

        start_index = full_text.find(first_line)
        if start_index == -1:
            logger.warning("Could not locate first line for chapter %r; skipping.", title)
            continue

        if i + 1 < len(markers):
            next_first_line = markers[i + 1]["first_line"]
            end_index = full_text.find(next_first_line)
            if end_index == -1:
                end_index = len(full_text)
        else:
            end_index = len(full_text)

        chapter_content = full_text[start_index:end_index].strip()
        chapters.append({"title": title, "content": chapter_content})

    return chapters


def save_chapters(book_name: str, chapters: list, base_folder: str = BASE_PARSED_FOLDER) -> None:
    """Persist each chapter as a separate ``.txt`` file.

    Args:
        book_name: Name used as the sub-folder inside *base_folder*.
        chapters: List of dicts with ``"title"`` and ``"content"`` keys.
        base_folder: Root directory where parsed books are stored.
                     Defaults to the ``PARSED_BOOKS_FOLDER`` env var or
                     ``"parsed_books"``.

    Raises:
        OSError: If the directory cannot be created or a file cannot be written.
    """
    safe_book_name = safe_filename(book_name)
    book_folder = os.path.join(base_folder, safe_book_name)

    try:
        os.makedirs(book_folder, exist_ok=True)
    except OSError:
        logger.exception("Failed to create directory: %s", book_folder)
        raise

    for chapter in chapters:
        try:
            title = safe_filename(chapter["title"])
        except ValueError:
            logger.warning("Skipping chapter with unsanitary title: %r", chapter.get("title"))
            continue

        # Guard against directory traversal
        file_path = os.path.realpath(os.path.join(book_folder, f"{title}.txt"))
        if not file_path.startswith(os.path.realpath(book_folder)):
            logger.error("Directory traversal attempt detected for title %r; skipping.", title)
            continue

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chapter["content"])
        except OSError:
            logger.exception("Failed to write chapter file: %s", file_path)
            raise

    logger.info("Chapters saved in %s", book_folder)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def extract_and_process_text(pdf_path: str) -> str:
    """Extract text from a PDF and log progress.

    Args:
        pdf_path: Path to the input PDF.

    Returns:
        Extracted (truncated) book text.
    """
    logger.info("Extracting text from %s …", pdf_path)
    return extract_text(pdf_path)


def identify_and_split_chapters(text: str) -> list:
    """Run AI chapter detection and split the text accordingly.

    Args:
        text: Extracted book text.

    Returns:
        List of chapter dicts with ``"title"`` and ``"content"`` keys.
    """
    logger.info("Identifying chapters using AI …")
    markers = get_chapter_markers(text)
    return split_book_locally(text, markers)


def save_extracted_chapters(book_name: str, chapters: list, base_folder: str = BASE_PARSED_FOLDER) -> None:
    """Persist the extracted chapters to disk.

    Args:
        book_name: Name of the book (used as folder name).
        chapters: List of chapter dicts.
        base_folder: Root folder for output.
    """
    logger.info("Saving %d chapters …", len(chapters))
    save_chapters(book_name, chapters, base_folder)
    logger.info("Done!")


def process_book(pdf_path: str, base_folder: str = BASE_PARSED_FOLDER) -> None:
    """End-to-end pipeline: extract → split → save.

    Args:
        pdf_path: Path to the input PDF file.
        base_folder: Root directory where parsed books are stored.
    """
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]

    text = extract_and_process_text(pdf_path)
    chapters = identify_and_split_chapters(text)
    save_extracted_chapters(book_name, chapters, base_folder)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _pdf_path = os.getenv(
        "INPUT_PDF",
        r"C:\Users\baps\sunita\Test\input_pdf\The Little Book of Good Thi_ (Z-Library).pdf",
    )
    process_book(_pdf_path)
