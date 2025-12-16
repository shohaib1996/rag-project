import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided information."
"""


def generate_answer(context: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
                Context: {context}
                Question: {question}
                """,
            },
        ],
        temperature=0,
        max_tokens=50,
    )

    return response.choices[0].message.content.strip()


def generate_title(question: str) -> str:
    """
    Generates a short 3-5 word title for a conversation based on the first question.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Generate a short, concise title (max 5 words) for the user's question. Do not include quotes.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.5,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()
