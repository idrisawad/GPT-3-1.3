# GPT-3⅓

GPT-3⅓ is a language generation model which has been trained on a diverse range of internet text, and can generate human-like text on a wide variety of topics. 

GPT-3 (Generative Pre-trained Transformer 3) is a state-of-the-art language generation model developed by OpenAI. It has been trained on a diverse range of internet text, and can generate human-like text on a wide variety of topics. It has been used to create chatbots, generate text for language translation, summarization, and question-answering, and can also perform tasks such as text completion, text classification, and text generation.

GPT-3⅓ is a language generation model which is  deep learning chatbot in Python that can learn from the internet and remember every sentence it has talked.
It has been trained on a diverse range of internet text, and can generate human-like text on a wide variety of topics, inspired, by and created of the methodology, of OpenAIs GPT-3. 

### 1. Overview ### 

The Deep Learning Chatbot is an AI-powered chatbot that uses GPT-3 to generate responses to user inputs. The chatbot can be fine-tuned using a dataset of text inputs and corresponding outputs. The chatbot can also learn from the internet and remember every sentence ever talked. 

### 2. Installation ### 

To use the Deep Learning Chatbot, you will need to install the following dependencies:

`nltk`: A library for natural language processing tasks such as tokenization and lemmatization.
`sklearn`: A library for machine learning tasks such as feature extraction and similarity measurement.
`transformers`: A library for state-of-the-art natural language processing models such as GPT-3.
`torch`: A library for deep learning tasks such as model fine-tuning and training.

You can install these dependencies by running the following command:
```
pip install -r requirements.txt
```

To install the necessary libraries and dependencies for the script, and initialize the AI for the first time, you can use the following command:

```
pip install -r requirements.txt && python -c "from deep_learning_chatbot import DeepLearningChatbot; dlc = DeepLearningChatbot(); dlc.fine_tune_gpt3('model_name', 'dataset_path', batch_size=4, device='cuda')"
```

It's important to note that `model_name` should be one of the models available in the transformers library, for example : "distilgpt2" , "gpt2" or "openai-gpt" and dataset_path should be the path to your dataset where you want to fine-tune the model.

This command will first install the necessary libraries and dependencies using pip, and then it will initialize the AI by creating an instance of the `DeepLearningChatbot` class and fine-tune the GPT-3 model using the specified dataset.

### 3. Usage ### 

#### 3.1 Initialization #### 

To use the chatbot, you need to create an instance of the DeepLearningChatbot class:

```chatbot = DeepLearningChatbot()```

#### 3.2 Learning from the Internet #### 
To learn from the internet, you can use the learn_from_internet method, which takes a string of text as a parameter:

text = "This is an example of text that the chatbot can learn from the internet"
chatbot.learn_from_internet(text)

#### 3.3 Fine-tuning the GPT-3 model #### 
To fine-tune the GPT-3 model, you can use the fine_tune_gpt3 method, which takes the following parameters:

model_name: The name of the GPT-3 model to use (e.g., "distilgpt2").
train_texts: A list of texts to use for fine-tuning the model.
train_labels: A list of labels corresponding to the texts.
batch_size: The number of samples to use in each training iteration.
device: The device to use for training (e.g., "cuda" for a GPU).
```
model_name = "distilgpt2"
train_texts = ["This is an example of text that the chatbot can learn from","This is an example of text that the chatbot can learn from"]
train_labels = [0,1]
batch_size = 4
device = "cuda"
chatbot.fine_tune_gpt3(model_name, train_texts, train_labels, batch_size, device)
```

#### 3.4 Chatting #### 

To start chatting with the chatbot, you can use the chat method:

```chatbot.chat()```

This will start a loop where the user can input a sentence, and the chatbot will generate a response. The loop will continue until the user inputs "bye".

3.5 Saving and loading the fine-tuned model

The chatbot allows you to save and load the fine-tuned model, optimizer, and scheduler state, respectively.
You can use the save_model method which takes the path where the model should be saved, and load_model takes the path where the model is saved.

```
path = "path/to/save/model"
chatbot.save_model(path)
```
and to load the model:
```
path = "path/to/save/model"
chatbot.load_model(path)
```

In the chat method, instead of using the Tf-Idf similarity to find the most similar sentence, I've used the fine-tuned GPT-3 model to generate a response.

The Code also use the method called _normalize that is used to normalize the user's input and the sentences. This method is used to clean the text by lowercasing it, removing punctuation and tokenizing it.

Here is a detailed preparation guide with code hints to run GPT-3-1.3 Deep Learning Chatbot code properly:

Download the necessary NLTK data:

```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```
Load the fine-tuned GPT-3 model:

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai/gpt3-squad2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

Prepare the data for fine-tuning, this step is optional if you don't want to fine-tune the model:

train_texts = [ "code snippet 1 in language A", "code snippet 2 in language B", ...]
train_labels = [ "language A", "language B", ...]
```

Initialize the chatbot:

```chatbot = DeepLearningChatbot()
chatbot.fine_tune_gpt3(model_name, train_texts, train_labels, batch_size=4, device='cuda')
```

Use the chatbot to generate code snippets in different programming languages:

code_snippet = chatbot._get_response("Write a function that takes two integers as input and returns their sum in Python")

To check if the code generated is correct, you can use a linter or a formatter:

```
from py_linter import PyLinter
linter = PyLinter()
linter.lint(code_snippet)
```

Please note that this is a very high-level guide, and there may be additional steps and considerations that need to be taken depending on your specific use case.
Also, make sure you have a good understanding of the programming languages and tools that you are using.

Here is a single command line that can be used to install the required libraries, download the necessary NLTK data, load the fine-tuned GPT-3 model, and initialize the chatbot:

```
pip install nltk scikit-learn transformers torch; python -c "import nltk;nltk.download('punkt');nltk.download('wordnet');from transformers import AutoModelForCausalLM, AutoTokenizer;model=AutoModelForCausalLM.from_pretrained('openai/gpt3-squad2');tokenizer = AutoTokenizer.from_pretrained('openai/gpt3-squad2');from deep_learning_chatbot import DeepLearningChatbot;chatbot = DeepLearningChatbot();chatbot.fine_tune_gpt3('openai/gpt3-squad2', train_texts, train_labels, batch_size=4, device='cuda')"
```

This command will install the required libraries, download the necessary NLTK data, load the fine-tuned GPT-3 model, and initialize the chatbot.

It's worth mentioning that this code is an example, and it might need further adjustments and fine-tuning to work correctly. Moreover, the fine-tuning process might take a while depending on the size of your dataset and the computational resources you have.

### 4.FAQ ### 

#### 4.1 How to create training data texts? ####

There are several ways to create training data texts for fine-tuning the GPT-3 model in the Deep Learning Chatbot code. Here are a few examples:

**Scraping:** You can use web scraping techniques to gather code snippets from websites such as Stack Overflow, GitHub, and other code repositories.

**Manual Creation:** You can manually create code snippets in different programming languages. This can be time-consuming but it guarantees that the data is relevant to your task and that it is of high quality.

**Crowdsourcing:** You can use platforms such as Amazon Mechanical Turk or Upwork to outsource the creation of the training data to a large number of people.

**Synthetic Data:** You can use a code generator to create synthetic data. This is useful if you have a specific type of code or a specific programming language in mind.

**Open-source Datasets:** There are datasets that are available on-line like https://github.com/mhagiwara/deep_learning_code_completion

*Mix of the above: You can use a combination of the above methods to create a diverse and high-quality training dataset.*

*It's important to ensure that the data is relevant to your task, that it is of high quality, and that it is diverse in terms of programming languages and code types. Also, you should consider the size of the dataset, the larger the dataset, the better the model will perform.*

#### 4.2 How to fine-tune? #### 

Fine-tuning a pre-trained language model like GPT-3 involves training the model on a new dataset for a specific task. The process can be broken down into the following steps:

**Data preparation:** The first step is to prepare the data for fine-tuning. This includes cleaning and pre-processing the data, such as tokenizing the text, lowercasing it, and removing punctuation. You will also need to split your data into training and validation sets.

**Model instantiation:** You need to instantiate the pre-trained model and tokenizer from the transformers library. You can do this by specifying the model's name and loading the pre-trained weights.

**Data loading:** The next step is to load the data into the model for training. You need to convert the data into the format that the model expects, which is typically a tensor. This is done by tokenizing the text, and encoding it into numerical values.

**Fine-tuning:** Once the data is loaded, you can fine-tune the model by training it on the new data. The fine_tune method of the transformers library can be used for this step. The method takes the input_ids (the encoded text) and labels as input. You can set the number of training epochs and batch size as well.

**Evaluation:** After the fine-tuning is complete, you should evaluate the model's performance on the validation set. You can use metrics such as accuracy, precision, recall, and F1-score to evaluate the model's performance.

**Hyperparameter tuning:** You can fine-tune the model further by tuning the hyperparameters such as learning rate, batch size, and number of training epochs. This can help you to find the best set of hyperparameters for your specific task.

**Saving and loading the model:** Once you are satisfied with the performance of the fine-tuned model, you should save it for future use. You can save the model and tokenizer using the save_pretrained method of the transformers library. You can then use the from_pretrained method to load the model and tokenizer.

*Please note that fine-tuning a pre-trained language model can be computationally intensive and requires a powerful GPU, and a lot of memory. It's also important to mention that fine-tuning a pre-trained model requires a good amount of data and computational resources, and you should consider this when deciding on whether to use the OpenAI API or to train your own model.*

#### 4.2 How to train this model? #### 

**Prepare a dataset:** Gather a large dataset of text data that is relevant to the task you want the chatbot to perform. This dataset should include text data that the chatbot can learn from and text data that you can use to evaluate the model's performance.

**Fine-tune the GPT-3 model:** Use the fine_tune_gpt3() method provided in the code to fine-tune the GPT-3 model on your dataset. This method takes the model name, train texts, and train labels as input.

**Evaluate the model:** Evaluate the performance of the fine-tuned model on a separate dataset. You can use metrics such as accuracy, precision, recall, and F1-score to evaluate the model's performance.

**Hyperparameter tuning:** Try different hyperparameters such as learning rate, batch size, and number of training epochs to find the best set of hyperparameters for your specific task.

**Saving the model:** Once you are satisfied with the performance of the fine-tuned model, you should save it for future use. You can use the save_pretrained method of the transformers library to save the model and tokenizer.

#### 4.4 What means device cuda? #### 

In the context of this script, the device parameter specifies which device the model and data should be loaded on to perform the computation.

The value 'cuda' means that the script will use the GPU (if available) to perform the computation. CUDA (Compute Unified Device Architecture) is a parallel computing platform and API developed by NVIDIA that allows using the GPU for general purpose computing. If a GPU with CUDA support is available, it will allow the model to perform computation much faster than using a CPU.

The value 'cpu' means that the script will use the CPU to perform the computation.

It's important to note that in order to use the 'cuda' option, your system must have a NVIDIA GPU with CUDA support and you must have the CUDA toolkit and NVIDIA drivers installed on your system.


#### 4.4 What means batch size? #### 

In the context of this script, the batch_size parameter determines the number of samples in a single batch of data that is passed through the model at once during training.

A larger batch size allows the model to make better use of the GPU and perform more computation in parallel, which can lead to faster training times. However, it also requires more memory to store the activations and gradients.

A smaller batch size allows the model to use less memory, but it will also result in slower training times.

When you fine-tune a model, you will generally want to use a batch size that is as large as possible while still fitting into the GPU memory. However, it also depends on the specific dataset and the size of the model, so you may need to experiment with different batch sizes to find the best value for your use case.

It's also worth noting that in general a batch size of 4 is a good starting point for fine-tuning GPT-3 models, but you may need to increase or decrease it depending on the specific dataset and the GPU memory available.

### 5. Troubleshooting #### 

 - If the chatbot is not generating good responses, it might be because the fine-tuning process did not converge, or the dataset was not adequate.
 - If the chatbot is not learning from the internet, it might be because the text passed to the learn_from_internet method is not adequate.
 - If the code is not running correctly, it might be because the dependencies are not correctly installed, or the versions are not compatible.

In this code, I have used the HuggingFace's transformers library to fine-tune a pre-trained GPT-3 model on the chatbot's learned data. I've added a new method fine_tune_gpt3 that takes the model name, train texts and train labels as input. This method instantiates the GPT-3 model and tokenizer, then it prepares the data for fine-tuning and fine-tune the model.


### 6. Conclusion #### 

The Deep Learning Chatbot is an AI-powered chatbot that uses GPT-3 to generate responses to user inputs. The chatbot can be fine-tuned using a dataset of text inputs and corresponding outputs. The chatbot can also learn from the internet and remember every sentence ever talked. With the above-mentioned instructions, you should be able to use the chatbot effectively.

#### 6.1 The potentials of this code are: ###

 - **Customization:** The chatbot can be fine-tuned using a dataset of text inputs and corresponding outputs, which allows for customization to the specific needs of the business.

 - **Natural Language Processing:** The use of GPT-3, a state-of-the-art natural language processing model, allows the chatbot to generate human-like responses, which can improve the user experience.

 - **Scalability:** The chatbot can learn from the internet, which allows it to scale and adapt to new situations and conversations.

 - **Cost-effective:** Chatbots can be more cost-effective than human customer service representatives.

 - **24/7 Availability:** Chatbots can be available 24/7, which can improve customer satisfaction and engagement.

 - **Multilingual:** The Chatbot can be fine-tuned to support multiple languages.

 - **Memory:** The Chatbot can remember every sentence ever talked, which can improve its ability to understand the context of the conversation.

 - **Healthcare:** The chatbot can be used in healthcare industry to assist patients in remote diagnosis, monitoring and treatment.

 - **E-commerce:** The chatbot can be used in e-commerce industry to assist customers with product recommendations, order tracking and purchase assistance.

 - **Customer service:** The chatbot can be used in customer service industry to assist customers with common queries, complaints and feedback.

However, it's important to note that this code is an example, and in order to turn it into a working product, there's more work to be done such as testing, fine-tuning, integration with other systems and scaling.


### 7. Legal Disclaimer: ###

The information provided on this GitHub page (the "Page") is for general informational purposes only. The Page is not intended to provide legal advice or create an attorney-client relationship. You should not act or rely on any information on the Page without seeking the advice of a qualified attorney. The developer(s) of this Page do not warrant or guarantee the accuracy, completeness, or usefulness of any information contained on the Page and will not be liable for any errors or omissions in the information provided or for any actions taken in reliance on the information provided.

#### 7.1 Policy: ####
All code and other materials provided on this Page are the property of the developer(s) and are protected by copyright and other intellectual property laws. You may not use, reproduce, distribute, or create derivative works from the code or materials on the Page without the express written consent of the developer(s). If you would like to use any of the code or materials provided on this Page for any purpose, please contact the developer(s) for permission.

The developer(s) reserve the right to make changes to the Page and to the code and materials provided on the Page at any time and without notice. The developer(s) also reserve the right to terminate access to the Page or to any code or materials provided on the Page at any time and without notice.

#### 7.2 Copyright Notice: #### 
Copyright (c) 2023 Idris Awad. All rights reserved. Any code or other materials provided on this Page are the property of the developer(s) and are protected by copyright and other intellectual property laws. Unauthorized use, reproduction, distribution, or creation of derivative works is prohibited.

Please note that using, reproducing or distributing the code or materials provided on this Page without proper attribution and without obtaining express permission from the developer(s) may result in copyright infringement and legal action being taken against you.

By accessing and using this Page, you acknowledge and agree to the terms of this legal disclaimer, policy and copyright notice.
