import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone

# Add backend directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()


def reset_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found.")
        return

    pc = Pinecone(api_key=api_key)
    index_name = "rag-app"
    index = pc.Index(index_name)

    print(f"⚠️  WARNING: This will delete ALL vectors in the index '{index_name}'.")
    confirm = input("Are you sure? (type 'yes' to confirm): ")

    if confirm.lower() == "yes":
        try:
            index.delete(delete_all=True)
            print("✅ distinct index cleared successfully.")
        except Exception as e:
            print(f"❌ Error deleting index: {e}")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    reset_pinecone()
