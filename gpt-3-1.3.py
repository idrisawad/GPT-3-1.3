import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

class DeepLearningChatbot:

    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        self.sentences = []

    def learn_from_internet(self, text):
        self.sentences = nltk.sent_tokenize(text)

    def _normalize(self, text):
        return nltk.word_tokenize(text.lower().translate(self.remove_punct_dict))

    def _lemmatize(self, word):
        return self.lemmatizer.lemmatize(word)

    def _get_response(self, user_input):
        # Vectorize the user's input and the sentences
        TfidfVec = TfidfVectorizer(tokenizer=self._normalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sentences + [user_input])
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf==0):
            return "I am sorry! I don't understand you"
        else:
            return self.sentences[idx]

    def chat(self):
        user_input = input("You: ")
        while user_input != 'bye':
            response = self._get_response(user_input)
            print("Chatbot: ", response)
            self.sentences.append(user_input)
            user_input = input("You: ")
        print("Chatbot: Bye!")

#initialize the chatbot and learn from the internet
chatbot = DeepLearningChatbot()
chatbot.learn_from_internet("Insert here the text you want the chatbot to learn from the internet")

#start chatting
chatbot.chat()gpt
