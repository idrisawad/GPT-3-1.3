import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.optim import Optimizer
import scrapy
from gtts import gTTS

class DeepLearningChatbot:

    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        self.sentences = []
        self.model = None
        self.tokenizer = None
        self.history = []
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.device = None
        self.batch_size = None

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

        tts = gTTS(response_text)
        tts.save('response.mp3')
        return response_text

    def load_dataset(self, file_path):
        texts = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                data = line.strip().split("\t")
                texts.append(data[0])
                labels.append(data[1])
        return texts, labels

    def fine_tune_gpt3(self, model_name, dataset_path, batch_size=4, device='cuda'):
    self.batch_size = batch_size
    self.device = device


    # Instantiate the model and tokenizer
    self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Freeze some layers
    for name, param in self.model.named_parameters():
        if 'layer.' not in name:
            param.requiresGrad = False

    # Prepare the data for fine-tuning
    train_texts, train_labels = self.load_dataset(dataset_path) # <--- Fill the dataset path here
    input_ids = torch.tensor([self.tokenizer.encode(text) for text in train_texts])
    labels = torch.tensor(train_labels)

    # Initialize the optimizer and scheduler
    self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=5e-5)
    self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
    self.loss_fn = nn.CrossEntropyLoss()

    # Fine-tune the model
    for epoch in range(10):
        for i in range(0, len(input_ids), batch_size):
            input_batch = input_ids[i:i + batch_size].to(device)
            labels_batch = labels[i:i + batch_size].to(device)

            self.optimizer.zero_grad()
            logits = self.model(input_batch)[0]
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels_batch.view(-1))
            loss.backward()
            self.optimizer.step()

        self.scheduler.step(loss)

class CodeSpider(scrapy.Spider):
    name = "code"
    start_urls = [
        'https://stackoverflow.com/questions/tagged/python',
        'https://github.com/search?q=python+code+snippet&type=Code'
    ]
    def parse(self, response):
        for code_snippet in response.css('pre'):
            yield {'code': code_snippet.css('::text').get()}
            with open("code_snippets.txt", "a") as f:
                f.write(code_snippet.css('::text').get() + "\n")
