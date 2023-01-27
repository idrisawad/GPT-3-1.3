import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

nltk.download('punkt')

class DeepLearningChatbot:

    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        self.sentences = []
        self.model = None
        self.tokenizer = None

    def learn_from_internet(self, text):
        self.sentences = nltk.sent_tokenize(text)

    def _normalize(self, text):
        return nltk.word_tokenize(text.lower().translate(self.remove_punct_dict))

    def _lemmatize(self, word):
        return self.lemmatizer.lemmatize(word)

    def _get_response(self, user_input):
        # Use the fine-tuned GPT-3 model to generate a response
        input_text = f"{user_input} <stop>"
        input_ids = torch.tensor(self.tokenizer.encode(input_text)).unsqueeze(0)
        response = self.model.generate(input_ids)[0]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)

        return response_text

    def fine_tune_gpt3(self, model_name, train_texts, train_labels):
        # Instantiate the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare the data for fine-tuning
        input_ids = torch.tensor([self.tokenizer.encode(text) for text in train_texts])
        labels = torch.tensor(train_labels)

        # Fine-tune the model
        self.model.fine_tune(input_ids, labels)

    def chat(self):
        user_input = input("You: ")
        while user_input != 'bye':
            response = self._get_response(user_input)
            print("Chatbot: ", response)
            self.sentences.append(user_input)
            user_input = input("You: ")
        print("Chatbot: Bye!")

#initialize the chatbot
chatbot = DeepLearningChatbot()

# Learn from the internet
chatbot.learn_from_internet("Insert here the text you want the chatbot to learn from the internet")

# Fine-tune GPT-3 on the chatbot's learned data
chatbot.fine_tune_gpt3("gpt3", chatbot.sentences, [0] * len(chatbot.sentences))
#start chatting
chatbot.chat()
