# Twitter Sentiment Analysis with a Bidirectional LSTM

A deep learning project that classifies tweets into four sentiment categories using a bidirectional LSTM built in TensorFlow/Keras.

## Goal
Given the text of a tweet, predict whether its sentiment is **Positive**, **Negative**, **Neutral**, or **Irrelevant**.

## Data
- **Source:** Twitter entity sentiment dataset (originally from Kaggle), loaded from the [DeepDive-Into-AI-Workshop](https://github.com/KrishnaRohithVemulapalli/DeepDive-Into-AI-Workshop) GitHub repo
- Each row has a tweet, the company or game it mentions, and a sentiment label
- **Cleaning:**
  - Merged the training and validation files, since the validation file was small
  - Removed 2,700 duplicate rows and 686 rows with missing text
  - Dropped the company column so the model learns from the tweet text itself, not from which company is mentioned
- About **72,000 tweets** after cleaning, split randomly into 80% training and 20% testing (14,428 test tweets)

## Text preprocessing
Each tweet was cleaned with NLTK:
1. Removed special characters, numbers, and single letters
2. Converted to lowercase
3. Split into words (tokenization)
4. Lemmatized words (e.g., "cats" → "cat")
5. Removed stopwords (common words like "the" and "and") and words with 3 or fewer letters
6. Removed repeated words within the same tweet

Then I used the Keras Tokenizer to turn words into numbers (keeping the 20,000 most common words out of 24,130) and padded every tweet to 100 tokens.

## Model
| Layer | Details |
|---|---|
| Embedding | 100-dimensional word vectors, learned during training |
| Dropout | 50%, to reduce overfitting |
| Bidirectional LSTM | 150 units, reads the tweet forward and backward |
| Dense | 32 units, ReLU |
| Output | 4 units, softmax (one per sentiment class) |

- **Optimizer:** Adam (learning rate 0.0001)
- **Loss:** categorical cross-entropy
- **Training:** 20 epochs

**Tools:** Python, TensorFlow/Keras, NLTK, scikit-learn, pandas, NumPy, matplotlib, seaborn

## Results
**Test accuracy: 86.1%** (training accuracy was 90.8%, so there's some overfitting, but not much)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Irrelevant | 0.88 | 0.82 | 0.85 |
| Negative | 0.90 | 0.86 | 0.88 |
| Neutral | 0.88 | 0.86 | 0.87 |
| Positive | 0.80 | 0.89 | 0.84 |

The model performed fairly evenly across all four classes. Negative tweets were classified best, and Positive had the lowest precision, meaning some tweets from other classes were incorrectly labeled Positive.

## Limitations
- **Near-duplicate tweets may inflate the accuracy.** The dataset has many versions of the same tweet with small wording changes. I only removed exact duplicates, so similar versions of the same tweet can end up in both the training and test sets, making the test easier than it should be. The accuracy on completely new tweets would likely be lower.
- **The preprocessing removes words that matter for sentiment.** NLTK's stopword list includes negations like "not" and "no," and removing words with 3 or fewer letters drops words like "bad." For example, "not good" becomes just "good."
- **The test set was also used as validation data during training.** I didn't use it to pick the model, since training ran for a fixed 20 epochs, but a separate validation set would be cleaner.
- **The "Irrelevant" label is vague**, since it's not really a sentiment, which may make it harder to learn.

## Next steps
- Remove near-duplicate tweets before splitting, or split by original tweet, for a more honest accuracy
- Keep negation words like "not" and "no" during preprocessing
- Use a separate validation set and early stopping
- Try pretrained word embeddings (like GloVe) or a transformer model (like BERT) and compare

## How to run it
1. Open `LSTMTwitterSentimentalAnalysis.ipynb` in Google Colab.
2. Turn on a GPU (Runtime → Change runtime type → GPU). Training takes about 10 minutes.
3. Run all cells. The data loads automatically from GitHub.

## Files
- `LSTMTwitterSentimentalAnalysis.ipynb`: the full notebook (preprocessing, model, training, and evaluation)
