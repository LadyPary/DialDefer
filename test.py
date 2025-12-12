import ollama
print("test script")
resp = ollama.chat(
    model="llama3.2:3b-50k",
    messages=[{"role": "user", "content": "Say one short sentence about conflict."}],
)

print(resp["message"]["content"])