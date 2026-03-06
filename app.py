from google import genai
api_key = ""
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
    
    prompt =genai.ContentPrompt("Explain how AI works in a few words")

)

print(response.text)