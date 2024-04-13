# Integrating Large Language Models into Verbal Autopsy Workflows

This code is the culmination to my Capstone project at the University of Washington: Integrating Large Language Models in Verbal Autopsy Workflows.
The project is built on the transformer platform from huggingface and utlises three different AI tasks, Speech-to-text, Zero-shot classification, and text generation.

* Speech-to-text using  [Whisper](https://huggingface.co/openai/whisper-large-v3)  from OpenAI
* Text generation multiple options but most robustly tested using [ChatGPT4](https://openai.com/gpt-4) with the OpenAI api
* Zero-shot classification using [Bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) from Facebook

S-T-T is geared towards aiding VA administrators in the process of transciption and translation of interviews in near real time. Using the transcibed narrative, text generation and zero-shot classification approaches are used to try and classify a cause of death. By using the transformer platform, there is malleability in which models are chosen for classification allowing for new models to be adopted easily.
