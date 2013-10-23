from collections import Counter

class NaiveBayesClassifier:

    def __init__(self, examples):
        self.examples = examples
        self.vocabulary = set(self.extract_features(examples))
        self.vocabulary_size = len(self.vocabulary)
        self.probabilities_for_target = {}
        self.probabilities_for_word = {}

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
            self.probabilities_for_target[target_value] = p_target_value
            self.probabilities_for_word[target_value] = {}
            text = self.get_text(documents)
            distinct_word_positions = self.get_text_diff_words_count(text)
            for word in self.vocabulary:
                word_count_in_text = self.occurrences_count(word, text)
                p_word_target_value = float(word_count_in_text + 1) / float(distinct_word_positions + self.vocabulary_size)
                self.probabilities_for_word[target_value][word] = p_word_target_value

    def word_positions(self, document):
        words = document.split()
        positions = []
        for index, word in enumerate(words):
            if word in self.vocabulary:
                positions.append(index)
        return positions

    def classify(self, document):
        word_positions = self.word_positions(document)
        words_in_doc = document.split()
        max_prob = 0
        result = ''
        for target_value in self.examples.keys():
            p_target_value = self.probabilities_for_target[target_value]
            prod = self.probabilities_for_word[target_value][words_in_doc[0]]
            word_positions.pop(0)
            for pos in word_positions:
                prod *= self.probabilities_for_word[target_value][words_in_doc[pos]]
            prob = p_target_value * prod
            if prob > max_prob:
                max_prob = prob
                result = target_value
        return result

