# from livekit import elevenlabs
from elevenlabs.client import ElevenLabs
import os
from helper import extract_text, get_chapter_markers, save_chapters, split_book_locally
eleven_client = ElevenLabs(
    api_key=os.getenv("ELEVEN_API_KEY")
)

def text_to_audio(text, output_path, voice_id=None):
    if not voice_id:
        voice_id = "CwhRBWXzGAHq8TQ4Fs17"  # default example voice

    audio = eleven_client.text_to_speech.convert(
        voice_id=voice_id,
        output_format="mp3_44100_128",
        text=text
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)


def split_text_into_chunks(text, max_chars=4000):
    chunks = []
    while len(text) > max_chars:
        split_index = text.rfind(".", 0, max_chars)
        print(split_index)
        if split_index == -1:
            split_index = max_chars
        chunks.append(text[:split_index+1])
        text = text[split_index+1:]
        print(f"text ... {text[:100]} ...")
    chunks.append(text)
    return chunks


def convert_chapters_to_audio(book_folder):
    for file in os.listdir(book_folder):
        if file.endswith(".txt"):
            txt_path = os.path.join(book_folder, file)
            audio_filename = file.replace(".txt", ".mp3")
            audio_path = os.path.join(book_folder, audio_filename)

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = split_text_into_chunks(text)

            print(f"Generating audio for {file}...")

            with open(audio_path, "wb") as audio_file:
                for chunk in chunks:
                    audio_stream = eleven_client.text_to_speech.convert(
                        voice_id="CwhRBWXzGAHq8TQ4Fs17",
                        output_format="mp3_44100_128",
                        text=chunk
                    )
                    for piece in audio_stream:
                        audio_file.write(piece)

            print(f"Saved: {audio_filename}")


def process_book_and_generate_audio(pdf_path):
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print("Extracting text...")
    text = extract_text(pdf_path)

    print("Getting chapter markers from OpenAI...")
    markers = get_chapter_markers(text)

    print("Splitting chapters locally...")
    chapters = split_book_locally(text, markers)

    print("Saving chapters...")
    save_chapters(book_name, chapters)

    book_folder = os.path.join("parsed_books", book_name)

    print("Generating audio files...")
    convert_chapters_to_audio(book_folder)

    print("All done!")


# async def main():
#     book_folder = "parsed_books/Atomic_Habits"
#     await convert_chapters_to_audio(book_folder)

# if __name__ == "__main__":
#     asyncio.run(main())


if __name__ == "__main__":
    # pdf_path = r"C:\\Users\\baps\\sunita\\Test\\input_pdf\\The Little Book of Good Thi_ (Z-Library).pdf"
    # process_book_and_generate_audio(pdf_path)
    # split_text_into_chunks("This is a test. " * 300)
    convert_chapters_to_audio("parsed_books/The Little Book of Good Thi_ (Z-Library)")


