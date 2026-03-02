import boto3
import os


polly_client = boto3.client(
    "polly",
    region_name="us-east-1" 
)

def split_text_into_chunks(text, max_chars=2500):
    chunks = []
    while len(text) > max_chars:
        split_index = text.rfind(".", 0, max_chars)
        if split_index == -1:
            split_index = max_chars
        chunks.append(text[:split_index+1])
        text = text[split_index+1:]
    chunks.append(text)
    return chunks


def text_to_audio_polly(text, output_path, voice_id="Joanna"):
    chunks = split_text_into_chunks(text)

    with open(output_path, "wb") as audio_file:
        for chunk in chunks:
            response = polly_client.synthesize_speech(
                Text=chunk,
                OutputFormat="mp3",
                VoiceId=voice_id,
                Engine="neural"  # better quality
            )

            audio_stream = response["AudioStream"].read()
            audio_file.write(audio_stream)

def convert_chapters_to_audio(book_folder, voice_id="Aditi"):
    for file in os.listdir(book_folder):
        if file.endswith(".txt"):
            # txt_path = os.path.join(book_folder, file)
            list_of_files = [f for f in os.listdir(book_folder) if f.endswith('.txt')]
            print(f"list_of_files...{list_of_files}")
            txt_path = input()
            print(f"txt_path...{txt_path}")

            while True:
                print("\nAvailable Chapters:")
                for i, file in enumerate(list_of_files):
                    print(f"{i}. {file}")

                choice = input(
                    "\nEnter chapter number to convert (or type 'stop' to exit): "
                ).strip()

                if choice.lower() == "stop":
                    print("Stopping conversion.")
                    break

                if not choice.isdigit():
                    print("Invalid input. Please enter a number.")
                    continue

                index = int(choice) - 1

                if index < 0 or index >= len(list_of_files):
                    print("Not in list. Please choose a valid chapter number.")
                    continue


            selected_file = list_of_files[index]
            txt_path = os.path.join(book_folder, selected_file)
            audio_filename = txt_path.replace(".txt", ".mp3")

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            print(f"Generating audio for {txt_path}...")

            # text_to_audio_polly(text, audio_filename, voice_id=voice_id)
            # print(f"Saved: {audio_filename}")


            # try:
            #     text_to_audio_polly(text, audio_filename voice_id=voice_id)
            #     print(f"Saved: {audio_filename}")

            # except Exception as e:
            #     print("Polly Error:", e)

if __name__=="__main__":
    book_folder = r"C:\Users\baps\sunita\Test\parsed_books\The Little Book of Good Thi_ (Z-Library)11"
    convert_chapters_to_audio(book_folder, voice_id="Aditi")
