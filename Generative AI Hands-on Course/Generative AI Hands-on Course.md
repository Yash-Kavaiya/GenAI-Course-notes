# Generative AI Hands-on Course
###  Generative Al - Topics

1. Transformers
2.GPT models
3.Gen Al Apps
4.Gemini Apps
5. Open-source LLMS with HuaainaFace
6. Gradio Deployment
7. RAG
8. Ollama
9.Local RAG APP - Streamlit

## Transformer
A transformer is a neural network that learns the context of sequential data and generates new data out of it.
Transformers were first developed to solve the problem of sequence transduction, or neural machine
translation, which means they are meant to solve any task that transforms an input sequence to an output
sequence. This is why they are called "Transformers".

- Key Advantages of Transformer
 - Self Attention
 - Positional Encodig
 - Parallel Processin
 - Encoder - Decoder architecture

#### Limitations of RNN
Imagine an RNN as a person reading a sentence word by word, trying to understand the meaning as they go. As they read each word,
they remember a bit about what they've read so far-that's their memory. This memory helps them understand the next word better
and so on until the end of the sentence.
The challenge with RNNs is that they might forget earlier information if the sentence is too long, or they might become confused if there is too much information. LSTM & GRU rectify this to an extent

• Vanishing Gradient Problem
• Sequential Computation
• Difficulty Handling Long Dependencies
• Scalability
