- Use db.py to create and edit a vector database to store embeddings
- Use embed.py to create embeddings for given texts (format supported: .txt, .pdf, .py)
- Use qa.py to query the database for a given question and propagate the most similar answers plus question to a llm (in this case gpt) #TODO: add more llm's

!!! Further requirements:
Create secret_keys.py in the root directory with your own keys e.g.:
```python
OPENAI_API_KEY = "your_key"
```

Relevant links:
https://www.youtube.com/watch?v=wUAUdEw5oxM
(lost the rest of the links, will maybe add them later)