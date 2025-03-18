from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init
init(autoreset=True)
USER = Fore.YELLOW    
BO = Fore.MAGENTA    

corpus = [
    "hello",
    "bye",
    "how are you",
    "what is your name",
    "tell me about the weather",
    "default"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

responses = {
    "hello": "Hi there! How can I assist you?",
    "bye": "Goodbye! Have a great day!",
    "how are you": "I'm just a bot, but I'm doing fine!",
    "what is your name": "I'm your AI assistant!",
    "tell me about the weather": "I'm not sure, but you can check online for the latest weather updates.",
    "default": "I'm sorry, I don't understand that. Can you rephrase?"
}

def chatbot_ml_response(user_input):
    user_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([user_input])  
    similarity = cosine_similarity(input_vector, X)  

    index = similarity.argmax()  
    best_score = similarity[0, index]  
    if best_score < 0.3: 
        return "I'm sorry, I don't understand that. Can you rephrase?"

    return responses.get(corpus[index], responses["default"])

if __name__ == "__main__":
    print(BO + "Chatbot is running! Type 'exit' to stop.")
    while True:
        user_input = input(USER + "You: ") 
        if user_input.lower() == "exit":
            print(BO + "Chatbot: Goodbye!")
            break
        print(BO + "Chatbot:", chatbot_ml_response(user_input))
