from keras.datasets import imdb

training_set, testing_set = imdb.load_data(index_from = 3)

def display_entries(dataset, nums):
    word_index = imdb.get_word_index()

    reverse_word_index = {value + 3: key for key, value in word_index.items()}

    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'

    reviews, labels = dataset

    for n in nums:
        encoded_review = reviews[n]
        label = labels[n]

        english_review = ' '.join(reverse_word_index.get(i, '<UNK>') for i in encoded_review)

        sentiment = 'Positive' if label == 1 else 'Negative'

        print(f"Entry {n}:")
        print(f"Encoded: {encoded_review}")
        print(f"Review: {english_review}")
        print(f"Sentiment: {sentiment}")
        print()
    

entries = [6, 10, 159, 169, 190]

print("=== Training Set ===")
display_entries(training_set, entries)

print("=== Testing Set ===")
display_entries(testing_set, entries)
