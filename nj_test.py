# from neo4j import GraphDatabase
# from neo4j_graphrag.indexes import create_vector_index

# URI = "neo4j://localhost:7687"
# AUTH = ("neo4j", "Sunita")

# INDEX_NAME = "vector-index-name"

# # Connect to Neo4j database
# driver = GraphDatabase.driver(URI, auth=AUTH)

# # Creating the index
# create_vector_index(
#     driver,
#     INDEX_NAME,
#     label="Document",
#     embedding_property="vectorProperty",
#     dimensions=1536,
#     similarity_fn="euclidean",
# )
#------------------------------------------------------------------------

# import os

# folder_path = './parsed_books/The Little Book of Good Thi_ (Z-Library)' 
# total_word_count = 0

# # Loop through every file in the folder
# for filename in os.listdir(folder_path):
#     # Check if the file is a text file
#     if filename.endswith(".txt"):
#         file_path = os.path.join(folder_path, filename)
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 # Read content and split by whitespace to count words
#                 content = file.read()
#                 words = content.split()
#                 count = len(words)
                
#                 print(f"{filename}: {count} words")
#                 total_word_count += count
#         except Exception as e:
#             print(f"Could not read {filename}: {e}")

# print("-" * 20)
# print(f"Total words in folder: {total_word_count}")
#-----------------------------------------------------------------------

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
client = OpenAI()
speech_file_path = "speech.mp3"

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input="Today is a wonderful day to build something people love!",
    instructions="Speak in a cheerful and positive tone.",
) as response:
    response.stream_to_file(speech_file_path)