import unittest
import sure
from naive_bayes_classifier import NaiveBayesClassifier

class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.examples = {'university': ['''Abbottabad Public School , also commonly referred to as
        APS and Railway Public School , is a private , all boys , boarding
        school for , 7th to 12th grade students , located in Abbottabad ,
        Pakistan .''']}
        self.classifier = NaiveBayesClassifier(self.examples)

    def test_create_vocabulary(self):
        self.classifier.vocabulary.should.contain('private')

    def test_vocabulary_size(self):
        self.classifier.vocabulary_size.should.eql(28)

    def test_subset_of_documents_with_target_value(self):
        len(self.classifier.get_documents_with_target_value('university')).should.eql(1)

    def test_text_of_documents(self):
        documents = self.classifier.get_documents_with_target_value('university')
        self.classifier.get_text(documents).should.contain('private')

    def test_text_distinct_words(self):
        documents = self.classifier.get_documents_with_target_value('university')
        text = self.classifier.get_text(documents)
        self.classifier.get_text_diff_words_count(text).should.eql(28)

    def test_example_count(self):
        self.classifier.get_example_count().should.eql(1)

    def test_occurrences_of_word_count(self):
        documents = self.classifier.get_documents_with_target_value('university')
        text = self.classifier.get_text(documents)
        self.classifier.occurrences_count(',', text).should.eql(7)

    def test_learn(self):
        self.classifier.learn()

