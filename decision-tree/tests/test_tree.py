import unittest
import sure
from decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.tree = DecisionTree()
        self.data = [{"male": True, "tall": False, "rich": True, "married": False},
                {"male": False, "tall": True, "rich": True, "married": True},
                {"male": False, "tall": True, "rich": False, "married": True},
                {"male": True, "tall": True, "rich": True, "married": True},
                {"male": False, "tall": True, "rich": False, "married": True}]

    def test_entropy_corectness(self):
        entropy = self.tree.entropy(self.data, "married")
        is_within = entropy > 0.721 and entropy < 0.722
        is_within.should.be.ok

    def test_gain_corectness(self):
        gain = self.tree.gain(self.data, "male", "married")
        is_within = gain > 0.321 and gain < 0.322
        is_within.should.be.ok

    def test_find_best_attribute(self):
        attributes = list(self.data[0].keys())
        attributes.remove("married")
        best_attribute = self.tree.find_best_attribute(self.data,
                attributes, "married")
        best_attribute.should.eql("tall")

    def test_grow_decision_tree(self):
        attributes = list(self.data[0].keys())
        attributes.remove("married")
        root_node = self.tree.grow(self.data, attributes, "married")
        root_node.label.should.eql("tall")
