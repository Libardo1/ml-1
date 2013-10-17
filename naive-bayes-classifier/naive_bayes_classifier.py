from collections import Counter

class NaiveBayesClassifier:

    def __init__(self, examples):
        self.examples = examples
        self.vocabulary = set(self.extract_features(examples))
        self.vocabulary_size = len(self.vocabulary)

    def extract_features(self, examples):
        return [word for texts in examples.values() for text in texts for word
                in text.split()]

    def get_documents_with_target_value(self, target_value):
        return self.examples[target_value]

    def get_text(self, documents):
        text = ''
        for doc in documents:
            text += ' ' + doc
        return text

    def get_text_diff_words_count(self, text):
        return len(set(text.split()))

    def get_example_count(self):
        example_count = [len(texts) for texts in self.examples.values()]
        return sum(example_count)

    def occurrences_count(self, word, text):
        occurrences = Counter(text.split())
        return occurrences[word]

    def learn(self):
        for target_value in self.examples.keys():
            documents = self.get_documents_with_target_value(target_value)
            p_target_value = float(len(documents)) / float(self.get_example_count())
            text = self.get_text(documents)
            distinct_word_positions = self.get_text_diff_words_count(text)
            for word in self.vocabulary:
                word_count_in_text = self.occurrences_count(word, text)
                p_word_target_value = float(word_count_in_text + 1) / (distinct_word_positions + self.vocabulary_size)
