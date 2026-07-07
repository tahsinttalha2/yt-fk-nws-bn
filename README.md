# Bengali Fake News Detection from YouTube Videos
Social Media have improved our interpersonal communication and brought the world into the palm of our hands. But this advantage comes with a cost— a costly one. Information is spreading fast, a bit too fast. So fast that people rarely have the time to make an informed decision. There are already established methods to detect fake news without the consumer having to sweat researching the actual facts. But a tool that can detect fake news from YouTube links, for a language like Bengali, it pretty rare. This project evens the odds and makes it convenient to check fake news from YouTube videos. 

### How does it work then?
It follows a simple architecture:
* You give us the link, and we retrieve metadata about the video.
* If there is a good transcript, we use it. Otherwise we use our own ASR to generate one.
* We pass it through a transformer model fine-tuned to detect stylometric features of fake news.
* Then we pass the news through a RAG implemented LLM that evaluates the news based on facts.
* We use trusted sources to evaluate our verdict and then we show it to you.

### But why do we need it?
There are some pretty trusted fact checking organisation like BOOM Bangladesh, Rumour Scanner, etc. These require human-intervention to detect fake news. It's time consuming, requires a lot of background and in-person studies. We aim to reduce the gap. We want to give an informed decision that would accelerate productivity in this regard. Since we just use retrieval augmentation, our platform provides trusted and informed decision on the status of a news being fake, authentic or unconfirmed. 
