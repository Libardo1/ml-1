import math

class DecisionTree:

    def entropy(self, data, target_attribute):
        positive = 0.0
        negative = 0.0
        total_instances = float(len(data))
        for instance in data:
            if instance[target_attribute]:
                positive += 1.0
            else:
                negative += 1.0
        if positive == 0.0:
            positive_part = 0
        else:
            positive_part = - ((float(positive/total_instances)) *
                math.log(float(positive/total_instances), 2))
        if negative == 0.0:
            return positive_part
        negative_part = - ((negative/total_instances) *
                math.log(negative/total_instances, 2))
        return positive_part + negative_part

    def gain(self, data, attribute, target_attribute):
        target_entropy = self.entropy(data, target_attribute)
        possible_values = [True, False] 
        gain = target_entropy
        for val in possible_values:
            subset_data = [instance for instance in data if instance[attribute] == val]
            subset_val = float(len(subset_data) / len(data)) * self.entropy(subset_data, target_attribute)
            gain -= subset_val
        return gain

    def find_best_attribute(self, data, attributes, target_attribute):
        max_gain = 0
        best_attribute = attributes[0]
        for attribute in attributes:
            gain = self.gain(data, attribute, target_attribute)
            if gain > max_gain:
                best_attribute = attribute
                max_gain = gain
        return best_attribute
