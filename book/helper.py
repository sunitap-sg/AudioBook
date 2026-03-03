import logging
import os
import json
import re
import sys

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load environment variables early so every os.getenv() below picks them up
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: OpenAI model to use for chapter detection.
MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

#: Root folder where extracted chapters are written.
BASE_PARSED_FOLDER: str = os.getenv("PARSED_BOOKS_FOLDER", "parsed_books")

#: Number of characters sent to the LLM for chapter detection.
LLM_TEXT_LIMIT: int = 25_000

#: Maximum characters extracted from a PDF for initial processing.
PDF_EXTRACT_LIMIT: int = 3_000

#: Maximum characters captured from a table-of-contents section.
TOC_SECTION_LIMIT: int = 3_000

#: Maximum characters of extracted text logged at DEBUG level.
PDF_LOG_PREVIEW: int = 5_000

#: Maximum length allowed for a sanitized file/directory name.
FILENAME_MAX_LENGTH: int = 100

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Allow the caller to control verbosity via the LOG_LEVEL environment variable.
# Falls back to INFO (safer for production) if the variable is not set or
# contains an unrecognised level name.
_log_level_name: str = os.getenv("LOG_LEVEL", "INFO").upper()
_log_level: int = getattr(logging, _log_level_name, logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client initialisation
# ---------------------------------------------------------------------------

def _build_openai_client() -> OpenAI:
    """Create and return an authenticated OpenAI client.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable (set by
    :func:`load_dotenv` above).  Raises :class:`EnvironmentError` early rather
    than failing later with a cryptic API error.

    Returns:
        An initialised :class:`openai.OpenAI` client.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Add it to your .env file or export it before running this script."
        )
    return OpenAI(api_key=api_key)


client: OpenAI = _build_openai_client()


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
    """Sanitize a string so it can be used safely as a file-system name.

    Removes characters that are illegal on common file-systems (Windows and
    POSIX), replaces spaces with underscores, and truncates the result to
    :data:`FILENAME_MAX_LENGTH` characters.

    Args:
        name: The proposed file or directory name.

    Returns:
        A sanitized file name string no longer than :data:`FILENAME_MAX_LENGTH`
        characters.

    Raises:
        ValueError: If *name* is not a non-empty string, or if it becomes
                    empty after sanitization.
    """
    if not isinstance(name, str) or not name.strip():
        logger.warning("safe_filename received an empty or invalid input: %r", name)
        raise ValueError("File name must be a non-empty string.")

    # Remove characters that are illegal on Windows and POSIX file-systems.
    sanitized = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", name)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized[:FILENAME_MAX_LENGTH]

    if not sanitized.strip():
        logger.warning(
            "File name became empty after sanitization; original: %r", name
        )
        raise ValueError("File name is empty after sanitization.")

    return sanitized


def parse_json_from_llm_response(content: str) -> list:
    """Parse a JSON string that may be wrapped in markdown code fences.

    Strips markdown ``\`\`\`json`` / ``\`\`\``` fences that LLMs commonly add
    around their JSON output, then delegates to :func:`json.loads`.

    Args:
        content: Raw string returned by the LLM, potentially wrapped in
                 triple backticks.

    Returns:
        Parsed Python object (expected to be a list of chapter dicts).

    Raises:
        ValueError: If *content* is empty or blank.
        json.JSONDecodeError: If the content is not valid JSON after cleaning.
    """
    if not content or not content.strip():
        raise ValueError("LLM returned an empty response.")

    # Strip markdown code-fence wrappers (```json … ``` or plain ``` … ```).
    cleaned = re.sub(r"```json", "", content)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse JSON from LLM response. Error: %s\nRaw content:\n%s",
            exc,
            content[:500],
        )
        raise

    if not isinstance(result, list):
        raise ValueError(
            f"Expected a JSON array from the LLM, got {type(result).__name__}."
        )

    return result


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file and return a truncated preview.

    Only the first :data:`PDF_EXTRACT_LIMIT` characters are returned so that
    downstream callers (e.g. the LLM prompt builder) stay within sensible
    token limits.  The full text is not stored in memory beyond this function.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        The first :data:`PDF_EXTRACT_LIMIT` characters of concatenated page
        text.

    Raises:
        FileNotFoundError: If *pdf_path* does not point to an existing file.
        Exception: Re-raises any :mod:`pdfplumber` errors after logging them.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path!r}")

    pages_text: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
    except Exception:
        logger.exception("Error reading PDF: %s", pdf_path)
        raise

    full_text = "\n".join(pages_text)
    logger.debug(
        "First %d chars of extracted text:\n%s",
        PDF_LOG_PREVIEW,
        full_text[:PDF_LOG_PREVIEW],
    )
    logger.info(
        "Extracted %d total characters from %s.", len(full_text), pdf_path
    )
    return full_text[:PDF_EXTRACT_LIMIT]


def extract_contents_section(full_text: str) -> str | None:
    """Locate and return the table-of-contents block within *full_text*.

    Searches for a 'CONTENTS' or 'table of contents' heading and returns the
    text that follows it, up to :data:`TOC_SECTION_LIMIT` characters.

    Args:
        full_text: Raw text extracted from the PDF.

    Returns:
        Up to :data:`TOC_SECTION_LIMIT` characters of text starting after the
        heading, or ``None`` if no contents section is found.
    """
    pattern = r"(CONTENTS|table\s+of\s+contents)"
    match = re.search(pattern, full_text, re.IGNORECASE)
    logger.debug("Contents section match: %s", match)

    if not match:
        logger.info("No table-of-contents section found in the extracted text.")
        return None

    start = match.end()
    contents = full_text[start: start + TOC_SECTION_LIMIT]
    logger.debug("Contents section found:\n%s", contents)
    return contents


# ---------------------------------------------------------------------------
# Chapter detection — rule-based fallback
# ---------------------------------------------------------------------------

# Common chapter-heading patterns (case-insensitive).
_CHAPTER_HEADING_RE = re.compile(
    r"^(?:chapter\s+\w+[.:)–-]?\s*.+|part\s+\w+[.:)–-]?\s*.+|\d+[.:)]\s+.+)",
    re.IGNORECASE | re.MULTILINE,
)


def detect_chapters_by_rules(text: str) -> list[dict]:
    """Detect chapter boundaries using regular expressions (no LLM required).

    Looks for lines that match common chapter-heading patterns such as
    "Chapter 1: …", "Part I — …", or "1. Title".  The content of each chapter
    extends from its heading to the start of the next heading.

    Args:
        text: The full book text to analyse.

    Returns:
        A list of dicts with ``"title"`` and ``"first_line"`` keys, compatible
        with the output of :func:`get_chapter_markers`.  Returns an empty list
        if no chapter headings are found.
    """
    matches = list(_CHAPTER_HEADING_RE.finditer(text))
    if not matches:
        logger.warning("Rule-based chapter detection found no chapters.")
        return []

    chapters: list[dict] = []
    for i, match in enumerate(matches):
        title = match.group(0).strip()
        # The "first line" is the first non-empty line after the heading.
        rest_start = match.end()
        rest_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[rest_start:rest_end].lstrip()
        first_line = body.splitlines()[0].strip() if body else ""
        if first_line:
            chapters.append({"title": title, "first_line": first_line})
        else:
            logger.debug("Skipping heading with no body text: %r", title)

    logger.info(
        "Rule-based detector found %d chapter(s).", len(chapters)
    )
    return chapters


# ---------------------------------------------------------------------------
# Chapter detection — AI-assisted
# ---------------------------------------------------------------------------

def get_chapter_markers(book_text: str) -> list[dict]:
    """Use the OpenAI API to identify chapter titles and first sentences.

    Falls back to :func:`detect_chapters_by_rules` if the API call fails or
    returns no usable data.

    Args:
        book_text: Extracted book text.  Only the first :data:`LLM_TEXT_LIMIT`
                   characters are sent to the API.

    Returns:
        A list of dicts with ``"title"`` and ``"first_line"`` keys.

    Raises:
        RuntimeError: If both the LLM call and the rule-based fallback produce
                      no chapters.
    """
    prompt = (
        "The following text is from a book.\n\n"
        "Identify all chapters based on the content. "
        "Do not analyse the whole file if it has a contents page. "
        "For each chapter, return the title and the exact first sentence of the chapter.\n\n"
        "Return JSON in this exact format:\n"
        "[\n"
        "  {\n"
        '    "title": "Chapter Title",\n'
        '    "first_line": "Exact first sentence of chapter"\n'
        "  }\n"
        "]\n\n"
        "Only return valid JSON — no explanations, no markdown fences.\n\n"
        f"Book text:\n{book_text[:LLM_TEXT_LIMIT]}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw_content = response.choices[0].message.content
        logger.debug("Raw LLM response:\n%s", raw_content)
        markers = parse_json_from_llm_response(raw_content)
        if markers:
            return markers
        logger.warning("LLM returned an empty chapter list; trying rule-based fallback.")
    except Exception as exc:
        logger.warning(
            "LLM chapter detection failed (%s); falling back to rule-based detection.",
            exc,
        )

    # --- Rule-based fallback ---
    fallback_markers = detect_chapters_by_rules(book_text)
    if not fallback_markers:
        raise RuntimeError(
            "Neither the LLM nor the rule-based detector could identify any chapters."
        )
    return fallback_markers


# ---------------------------------------------------------------------------
# Chapter splitting and saving
# ---------------------------------------------------------------------------

def split_book_into_chapters(full_text: str, markers: list[dict]) -> list[dict]:
    """Split *full_text* into chapters using the provided chapter markers.

    Args:
        full_text: The complete book text.
        markers: List of dicts with ``"title"`` and ``"first_line"`` keys,
                 as returned by :func:`get_chapter_markers`.

    Returns:
        List of dicts, each containing ``"title"`` and ``"content"`` keys.
        Chapters whose ``first_line`` could not be located in the text are
        silently skipped with a warning.
    """
    chapters: list[dict] = []

    for i, marker in enumerate(markers):
        title = marker.get("title", "").strip()
        first_line = marker.get("first_line", "").strip()

        if not title or not first_line:
            logger.warning(
                "Marker at index %d is missing 'title' or 'first_line'; skipping.", i
            )
            continue

        start_index = full_text.find(first_line)
        if start_index == -1:
            logger.warning(
                "Could not locate first line for chapter %r; skipping.", title
            )
            continue

        # End of this chapter = start of the next chapter's first line.
        if i + 1 < len(markers):
            next_first_line = markers[i + 1].get("first_line", "")
            end_index = full_text.find(next_first_line) if next_first_line else -1
            if end_index == -1:
                end_index = len(full_text)
        else:
            end_index = len(full_text)

        chapter_content = full_text[start_index:end_index].strip()
        chapters.append({"title": title, "content": chapter_content})
        logger.debug("Split chapter %r (%d chars).", title, len(chapter_content))

    logger.info("Split %d chapter(s) from the book text.", len(chapters))
    return chapters


def save_chapters(
    book_name: str,
    chapters: list[dict],
    base_folder: str = BASE_PARSED_FOLDER,
) -> None:
    """Persist each chapter as a separate UTF-8 ``.txt`` file.

    Args:
        book_name: Human-readable book name used as the output sub-folder.
        chapters: List of dicts with ``"title"`` and ``"content"`` keys.
        base_folder: Root directory where parsed books are stored.
                     Defaults to :data:`BASE_PARSED_FOLDER`.

    Raises:
        ValueError: If *book_name* is invalid.
        OSError: If a directory cannot be created or a file cannot be written.
    """
    safe_book_name = safe_filename(book_name)
    book_folder = os.path.join(base_folder, safe_book_name)
    real_book_folder = os.path.realpath(book_folder)

    try:
        os.makedirs(real_book_folder, exist_ok=True)
    except OSError:
        logger.exception("Failed to create directory: %s", real_book_folder)
        raise

    saved_count = 0
    for chapter in chapters:
        try:
            title = safe_filename(chapter.get("title", ""))
        except ValueError:
            logger.warning(
                "Skipping chapter with unsanitizable title: %r",
                chapter.get("title"),
            )
            continue

        # Guard against directory-traversal attacks introduced by crafted titles.
        file_path = os.path.realpath(os.path.join(real_book_folder, f"{title}.txt"))
        if not file_path.startswith(real_book_folder + os.sep) and \
                file_path != real_book_folder:
            logger.error(
                "Directory traversal attempt detected for title %r; skipping.", title
            )
            continue

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chapter.get("content", ""))
            saved_count += 1
            logger.debug("Saved chapter file: %s", file_path)
        except OSError:
            logger.exception("Failed to write chapter file: %s", file_path)
            raise

    logger.info("Saved %d chapter file(s) in %s.", saved_count, real_book_folder)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def extract_book_text(pdf_path: str) -> str:
    """Extract and return text from a PDF, logging progress.

    Args:
        pdf_path: Path to the input PDF.

    Returns:
        Extracted (truncated) book text.
    """
    logger.info("Extracting text from %r …", pdf_path)
    text = extract_text(pdf_path)
    logger.info("Text extraction complete (%d chars).", len(text))
    return text


def detect_and_split_chapters(text: str) -> list[dict]:
    """Run chapter detection (AI with rule-based fallback) and split the text.

    Args:
        text: Extracted book text.

    Returns:
        List of chapter dicts with ``"title"`` and ``"content"`` keys.
    """
    logger.info("Detecting chapter boundaries …")
    markers = get_chapter_markers(text)
    logger.info("Detected %d chapter marker(s).", len(markers))
    return split_book_into_chapters(text, markers)


def persist_chapters(
    book_name: str,
    chapters: list[dict],
    base_folder: str = BASE_PARSED_FOLDER,
) -> None:
    """Write extracted chapters to disk, logging progress.

    Args:
        book_name: Name of the book (used as the output sub-folder).
        chapters: List of chapter dicts.
        base_folder: Root folder for output files.
    """
    logger.info("Saving %d chapter(s) for %r …", len(chapters), book_name)
    save_chapters(book_name, chapters, base_folder)
    logger.info("All chapters saved successfully.")


def process_book(pdf_path: str, base_folder: str = BASE_PARSED_FOLDER) -> None:
    """End-to-end pipeline: extract text → detect chapters → save files.

    Args:
        pdf_path: Path to the input PDF file.
        base_folder: Root directory where parsed books are stored.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        RuntimeError: If no chapters could be detected.
        OSError: If output files cannot be written.
    """
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info("Processing book: %r", book_name)

    text = extract_book_text(pdf_path)
    chapters = detect_and_split_chapters(text)
    persist_chapters(book_name, chapters, base_folder)

    logger.info("Finished processing %r.", book_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_pdf_path: str | None = os.getenv("INPUT_PDF")

    if not input_pdf_path:
        logger.error(
            "No input PDF specified. "
            "Set the INPUT_PDF environment variable to the path of the PDF file."
        )
        sys.exit(1)

    if not os.path.isfile(input_pdf_path):
        logger.error("Input PDF does not exist: %r", input_pdf_path)
        sys.exit(1)

    try:
        process_book(input_pdf_path)
    except Exception:
        logger.exception("An unexpected error occurred while processing the book.")
        sys.exit(1)
