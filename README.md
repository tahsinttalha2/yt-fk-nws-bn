# Bengali Fake News Detection from YouTube Videos

## Background & Inspiration
Social Media have improved our interpersonal communication and brought the world into the palm of our hands. But this advantage comes with a cost— a costly one. Information is spreading fast, a bit too fast. So fast that people rarely have the time to make an informed decision. There are already established methods to detect fake news without the consumer having to sweat researching the actual facts. But a tool that can detect fake news from YouTube links, for a language like Bengali, is pretty rare. This project evens the odds and makes it convenient to check fake news from YouTube videos. 

### How does it work then?
It follows a simple architecture:
* You give us the link, and we retrieve metadata about the video.
* If there is a good transcript, we use it. Otherwise, we use our own ASR to generate one.
* We pass it through a transformer model fine-tuned to detect stylometric features of fake news.
* Then we pass the news through a RAG implemented LLM that evaluates the news based on facts.
* We use trusted sources to evaluate our verdict, and then we show it to you.

### But why do we need it?
There are some pretty trusted fact checking organisation like BOOM Bangladesh, Rumour Scanner, etc. These require human-intervention to detect fake news. It's time-consuming, requires a lot of background and in-person studies. We aim to reduce the gap. We want to make an informed decision that would accelerate productivity in this regard. Since we just use retrieval augmentation, our platform provides trusted and informed decisions on the status of a news being fake, authentic or unconfirmed. 

## Model Architecture
It's time to discuss what our tool actually looks like underneath the hood. Here is a detailed architecture:

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/cbce703a-3d71-4c93-97ca-8840b5b19d8c" />

## Github Directory
- dataset/          -> contains the dataset used to train the detection model
- frontend/         -> contains the UI of the tool
- backend/          -> contains server, important backend work
- asr               -> automatic speech recognition tool
- detection-model   -> the model that analyses using stylometric features
- evaluation-model  -> the model that uses RAG to retrieve data and explain the decision
- README.md         -> the ultimate guide to this repository
- LICENSE           -> project license file
