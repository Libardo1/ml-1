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
