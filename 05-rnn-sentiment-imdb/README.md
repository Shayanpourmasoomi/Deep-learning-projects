# Sentiment Classification on IMDB: RNNs and LSTMs, Built-in and from Scratch

Recurrent models for binary sentiment classification of IMDB movie reviews. Each architecture is implemented twice, once with PyTorch's built-in module and once from scratch with custom cells, and the two families are compared.

## Models

| Model | Implementation |
|---|---|
| RNN | `nn.RNN`, and a custom RNN cell unrolled over time |
| LSTM | `nn.LSTM`, and a custom LSTM cell with input, forget and output gates |

All models share the same text pipeline: cleaning, tokenisation, vocabulary construction, padding and an embedding layer.

## Results (test set)

| Model | Test accuracy |
|---|---|
| LSTM (`nn.LSTM`) | **85.7%** |
| LSTM (from scratch) | 79.0% |
| Vanilla RNNs | Failed to learn a useful signal |

The contrast is the point of the exercise. Over long reviews, plain RNNs suffer from vanishing gradients and never learn, while the gated LSTM keeps information across hundreds of tokens. The notebook ends by scoring a new, unseen review with each model and discussing why their predictions differ.

## Files

```
rnn_lstm_imdb_sentiment.ipynb   # preprocessing, four models, training, evaluation, discussion
```

## Run

```bash
pip install torch datasets nltk scikit-learn pandas
```
