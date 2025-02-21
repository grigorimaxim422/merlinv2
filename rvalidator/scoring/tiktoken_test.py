import tiktoken

# Get the encoding for the GPT-3.5 turbo model
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Example: Encode a string
text = "Hello, world!"
encoded_text = encoding.encode(text)

print(f"Encoded Text: {encoded_text}")
