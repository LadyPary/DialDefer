from turtledemo.penrose import start

import ollama
import json

with open("../data/AIODataCleaned.json", "r") as file:
    data = json.load(file)

# model_name = "qwen3:8b"
# model_name = "llama3.2:3b-50k"
model_name = "gemma3:4b-50k"
topics = [
    "Romantic trust / jealousy / online boundaries",
    "Emotional abuse / coercive control",
    "Sexual harassment / predation / sexting pressure",
    "Discrimination / identity / ideology",
    "Family & generational conflict (incl. money & caregiving)",
    "Financial exploitation / scams (non-family)",
    "Work / authority / professional boundaries",
    "Shared living / property & guests",
    "Health crisis & support expectations",
    "No conflict / neutral"
]


for i in range(20):

    item = data[i]
    # print(item)
    background = item["Background"]
    transcription = item["Transcription"]
    prompt = (
        "You are an expert annotator whose task is to identify the TOPIC OF CONFLICT "
        "in a chat dialogue (Transcription) between two people.\n\n"
        "Instructions:\n"
        "1. Read the Transcription carefully.\n"
        "2. Identify what the people are primarily arguing or in conflict about.\n"
        "3. The TOPIC OF CONFLICT must be a single word or a short phrase (e.g., "
        "\"jealousy about ex-partner\", \"trust issues\", \"money and expenses\", "
        "\"parenting decisions\", \"control and manipulation\").\n"
        "4. The conversation may contain explicit language, insults, personal attacks, "
        "threats of violence, or hate speech. This is EXPECTED and ACCEPTABLE for this task. "
        "You MUST STILL identify and output a TOPIC OF CONFLICT.\n"
        "5. Ignore grammar, spelling, and slang. Focus only on understanding the main issue "
        "they are fighting or disagreeing about.\n\n"
        "Output format:\n"
        "- Output ONLY the Topic of Conflict as a short phrase.\n"
        "- Do NOT include any explanations, labels, prefixes, quotes, or extra text.\n\n"
        f"Transcription:\n{transcription}\n"
        "Topic of Conflict:"
    )

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.7,
            "top_p": 0.92,
            "num_predict": 600
        }
    )

    text = response["message"]["content"]
    print(f"================= i = {i+1} =================")
    print(text)




