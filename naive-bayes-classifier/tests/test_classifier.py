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
