import json
import ollama
# give data to chatgpt and ask it to categorize, not too specific (topic)
# then do all this
# use
# 1) Load your pooled seed examples (used as in-context examples)
with open("../data/pooledData.json", "r") as f:
    seeds = json.load(f)

def format_conversation_turns(turns):
    """turns = list of {speaker, text}"""
    lines = []
    for msg in turns:
        lines.append(f'{msg["speaker"]}: {msg["text"]}')
    return "\n".join(lines)

def buildPrompt(seeds, new_topic, new_background):
    topic_examples = [example for example in seeds if example["Topic"] == new_topic]

    prompt = []
    prompt.append(
        "You generate REALISTIC, conflict text dialogues between two people: Speaker A and Speaker B.\n"
        "You will be given three of the four items in an object. Using that, you need to generate a very realistic conversation (Transcription) between Speaker A and Speaker B.\n"
        "Each example that you generate has:\n"
        "1. SubmissionID: You can keep this as Synth_<alphanumeric string>, where alphanumeric string is a randomly generated string of length 10.\n"
        "2. Topic: One of the four values based on the input given to you: Emotional Abuse, Gaslighting, Boundary Issues, Manipulation and Control.\n"
        "3. Background: A short, one-line description describing the situation and providing context to the conversation.\n"
        "4. Transcription: A conversation transcript with labeled turns.\n"
    )
    prompt.append(
        "RULES:\n"
        "- Use ONLY Speaker A and Speaker B tags.\n"
        "- Make 7 to 10 turns per conversation.\n"
        "- Make the conflict emotionally realistic.\n"
        "- Definition: A SLANG is a type of language that consists of words and phrases that are regarded as very informal, are more common in speech than writing.\n"
        "- Definition: A TYPO is a typographical error, which is a mistake made during the process of typing or printing. It's a common type of error that occurs when fingers hit the wrong keys, letters are mixed up, or words are misspelled.\n"
        "- Chat data is NOT formal. KEEP the transcription informal. Use SLANG, don't use proper punctuation. Use TYPO sparsely.\n"
        "- Output ONLY the Transcription, no explanations.\n"
    )

    # Add all examples for this topic as in-context examples
    for index, example in enumerate(topic_examples, start=1):
        prompt.append(f"### Example {index}:\n")
        prompt.append(f"### Topic: {example['Topic']}\n")
        prompt.append(f"### Background: {example['Background']}\n")

        conv_obj = json.loads(example["Transcription"])
        turns = conv_obj["messages"]

        conv_text = format_conversation_turns(turns)

        prompt.append("Conversation:")
        prompt.append(conv_text)
        prompt.append("")  # blank line

    # New conversation to generate
    prompt.append("New Conversation that I want you to generate:")
    prompt.append(
        f"Topic: {new_topic}\n"
        f"Background: {new_background}\n"
        f"Conversation:\n"
        "Speaker A:"
    )
    return "\n".join(prompt)

def generateConversation(new_topic, new_background, model_name="qwen3:8b"):
    print(f"Generating conversation for topic='{new_topic}'...")
    prompt = buildPrompt(seeds, new_topic, new_background)
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
    return text

def parse_conversation(text):
    lines = text.strip().splitlines()
    messages = []
    for line in lines:
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        speaker = speaker.strip()
        msg = msg.strip()
        if speaker in {"Speaker A", "Speaker B"} and msg:
            messages.append({"speaker": speaker, "text": msg})
    return messages

def main():
    # 2) Load the 100 topic/background items you saved above
    with open("../data/AIODataCleaned.json", "r") as f:
        topic_background_items = json.load(f)

    output = []
    for i, item in enumerate(topic_background_items, start=1):
        new_topic = item["topic"]
        new_background = item["background"]

        raw_conv = generateConversation(new_topic, new_background)
        transcript = parse_conversation(raw_conv)

        out_obj = {
            "id": i,
            "topic": new_topic,
            "background": new_background,
            "transcript": transcript
        }
        output.append(out_obj)

    # 3) Save final synthetic dataset
    with open("../data/100SynthesizedData.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Saved 100SynthesizedData.json with", len(output), "items.")

if __name__ == "__main__":
    main()
