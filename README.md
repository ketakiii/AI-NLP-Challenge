# AI-NLP-Challenge

Problem Statement
Implement a deep learning model that learns to expand single variable polynomials, where the model takes the factorized sequence as input and predicts the expanded sequence. This is an exercise to demonstrate your machine-learning prowess, so please refrain from parsing or rule-based methods.

A training file is provided in S3:

train.txt: https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt
Each line of train.txt is an example, the model should take the factorized form as input, and predict the expanded form. E.g.

n*(n-11)=n**2-11*n
n*(n-11) is the factorized input
n**2-11*n is the expanded target
The expanded expressions are commutable, but only the form provided is considered as correct.


Reference link: https://keras.io/examples/nlp/lstm_seq2seq/

Character level recurrent sequence to sequence model: Encoder-Decoder Architecture

Summary of the algorithm

We start with input sequences from a domain and corresponding target sequences from another domain.
An encoder LSTM turns input sequences to 2 state vectors (we keep the last LSTM state and discard the outputs).
A decoder LSTM is trained to turn the target sequences into the same sequence but offset by one timestep in the future, a training process called "teacher forcing" in this context. It uses as initial state the state vectors from the encoder. Effectively, the decoder learns to generate targets[t+1...] given targets[...t], conditioned on the input sequence.
In inference mode, when we want to decode unknown input sequences, we:
Encode the input sequence into state vectors
Start with a target sequence of size 1 (just the start-of-sequence character)
Feed the state vectors and 1-char target sequence to the decoder to produce predictions for the next character
Sample the next character using these predictions (we simply use argmax).
Append the sampled character to the target sequence
Repeat until we generate the end-of-sequence character or we hit the character limit.
