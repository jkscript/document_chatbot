from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain, RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
import re
import dateparser
import os
import csv
import json
# Paste your huggingfacehub_api_token here
HUGGINGFACEHUB_API_TOKEN = ''

# Custom Prompt Template
custom_prompt_template = """Answer the user's question using the provided information. If you're unsure, refrain from guessing.

Context: {context}
Question: {question}

Provide helpful answers only. End your response with a courteous "Thank You."
"""

# Initialize Qdrant Client
url = 'http://localhost:6333'
collection_name = 'llm_vdb'
client = QdrantClient(url=url, prefer_grpc=False)

# Check and create collection if it doesn't exist
try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists.")
except UnexpectedResponse as e:
    if "Not found" in str(e):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Unexpected error: {e}")

# Validation Functions
def validate_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

def validate_phone(phone):
    return bool(re.match(r"^\+?\d{10,15}$", phone))

def parse_date(date_str):
    parsed_date = dateparser.parse(date_str)
    return parsed_date.strftime('%Y-%m-%d') if parsed_date else None

# Save user data to CSV
def save_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    headers = set(data.keys())

    if file_exists:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            existing_headers = next(reader, [])
            headers = headers.union(existing_headers)

    with open(filename, mode='w' if not file_exists else 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists or set(headers) != set(existing_headers):
            writer.writeheader()
        writer.writerow(data)

# Collect user information
def collect_user_info():
    user_info = {
        'name': input("Could you please share your name? "),
        'phone': collect_valid_input("May I have your phone number? ", validate_phone, "That doesn't seem like a valid phone number. Please try again."),
        'email': collect_valid_input("Could you provide your email address? ", validate_email, "That doesn't seem like a valid email format. Could you double-check it?")
    }
    print(f"Thank you, {user_info['name']}! We'll reach out to you at {user_info['phone']} or {user_info['email']}.")
    save_to_csv(user_info, filename="user_data.csv")
    return user_info

# Collect and validate input function
def collect_valid_input(prompt, validate_func, error_msg):
    while True:
        user_input = input(prompt)
        if validate_func(user_input):
            return user_input
        print(error_msg)

# Book appointment
def book_appointment():
    user_data = {
        'name': input("Please enter your name: "),
        'email': collect_valid_input("Please enter your email: ", validate_email, "Invalid email format. Please try again."),
        'phone': collect_valid_input("Please enter your phone number: ", validate_phone, "Invalid phone number format. Please try again."),
        'appointment_date': collect_valid_input("When would you like to book an appointment? (e.g., '2024-11-05' in YYYY-MM-DD format) ", parse_date, "Invalid date format. Please try again.")
    }
    print(f"Appointment booked for {user_data['name']} on {user_data['appointment_date']}. Thank you!")
    save_to_csv(user_data,filename="appointment_data.csv")
    return user_data

# Load intent mappings
def load_intent_mapping(filename="intent_mapping.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# Match user input to an intent
def get_intent(user_input, intent_mapping):
    user_input = user_input.lower()
    for intent, keywords in intent_mapping.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', user_input) for keyword in keywords):
            return intent
    return None

# Set custom prompts
def set_custom_prompts():
    return PromptTemplate(input_variables=['context', 'question'], template=custom_prompt_template, template_format='f-string')

# Load language model
def load_llm():
    print("Loading the language model from Hugging Face Hub...🚀")
    llm = HuggingFaceHub(
        repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model_kwargs={'temperature': 1, "max_length": 64, "max_new_tokens": 512}
    )
    print("Language model loaded successfully.")
    return llm

# Set up the retrieval-based QA chain
def retrival_qa_chain(llm, prompt, db):
    print("Setting up the retrieval-based QA chain...🔗")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    print("Retrieval-based QA chain ready.")
    return qa_chain

# Initialize QA bot
def qa_bot():
    print("Setting up the QA bot ...🤖")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = Qdrant(client=client, embeddings=embeddings, collection_name=collection_name)
    llm = load_llm()
    qa_prompt = set_custom_prompts()
    qa = retrival_qa_chain(llm, qa_prompt, db)
    print("QA bot ready.")
    return qa

# Main function
def main():
    intent_mapping = load_intent_mapping()
    qa_result = qa_bot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        intent = get_intent(user_input, intent_mapping)
        if intent == "callback":
            contact_info = collect_user_info()
            print("Callback Details:", contact_info)
        elif intent == "appointment":
            appointment_data = book_appointment()
            print("Appointment Details:", appointment_data)
        else:
            response = qa_result.invoke({'query': user_input})
            formatted_output = format_output(response)
            print("Bot:", formatted_output)

# Format output for user display
def format_output(answer):
    result = f"**Question:** {answer['query']} 🤔\n\n"
    result += f"**Answer:** 💡 {answer['result']} 🙏\n\n"

    if 'source_documents' in answer:
        result += "**Source Documents:**\n"
        for doc in answer['source_documents']:
            metadata = doc.metadata
            result += f"- Page {metadata['page']} from {metadata['source']}\n"
        result += "\n"

    result += 'This is only generated answer; for further official queries, please consult the concerned personnels.❗\n'
    return result

if __name__ == '__main__':
    main()
