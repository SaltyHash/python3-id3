#!/usr/bin/python3

# The MIT License
# 
# Copyright (c) 2016 Austin Bowen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Python3 implementation of the ID3 decision tree algorithm.
'''

__filename__ = 'id3.py'
__version__  = '0.1'
__author__   = 'Austin Bowen <arbowen@g.clemson.edu>'
__date__     = '2016-12-02 Fr'
__all__      = ['id3', 'ID3DecisionTree', 'ID3DecisionTreeNode']

from math import log2

class ID3DecisionTree:
    '''An ID3 decision tree.'''
    
    __slots__ = {'root_node': None}
    
    def __init__(self, root_node):
        self.root_node = root_node
    
    def __call__(self, inputs):
        '''The tree may be called like a function, returning the tree's output
        given a dict of inputs.
        '''
        node = self.root_node
        # Traverse the nodes until a leaf is reached
        while isinstance(node, ID3DecisionTreeNode):
            # Get the next node
            node = node.get_subnode_for_input(inputs[node.name])
        return node
    
    def __str__(self):
        '''Returns a "pretty" string showing the entire decision tree.
        Note:  If the recursion limit is hit, then there is probably a loop
        in the tree for some reason.
        '''
        
        def tree_str(node, prefix=''):
            if isinstance(node, ID3DecisionTreeNode):
                s = '{}{}:\n'.format(prefix, node.name)
                for value, subnode in node.value_subnode_pairs:
                    s += '{}{}'.format(prefix, value)
                    if isinstance(subnode, ID3DecisionTreeNode): s += '\n'
                    else: s += ' '
                    s += tree_str(subnode, prefix+'  ')
                return s
            else:
                return '-> {}\n'.format(node)
        
        return '{}:\n{}'.format(
            object.__str__(self),
            tree_str(self.root_node)[:-1]
        )
    
    def get_inputs_for_output(self, output):
        '''Returns a list of inputs that will give the specified output.'''
        if not isinstance(self.root_node, ID3DecisionTreeNode): return []
        
        def get_inputs(node, output, input={}, inputs=[]):
            for value, subnode in node.value_subnode_pairs:
                input[node.name] = value
                if isinstance(subnode, ID3DecisionTreeNode):
                    get_inputs(subnode, output, dict(input), inputs)
                elif (subnode == output):
                    inputs.append(dict(input))
            return inputs
        
        return get_inputs(self.root_node, output)
    
    def validate(self, exemplars):
        '''Returns True if the decision tree is valid in comparison to the given
        exemplars; that is, the tree is valid if its output matches the output
        for all (inputs, output) exemplar pairs.
        
        Note: The tree will be invalid if it was given duplicate, inconsistent
        exemplars (at least 2 exemplars with the same inputs but differing
        outputs) during training.
        '''
        for inputs, output in exemplars:
            if (self(inputs) != output): return False
        return True

class ID3DecisionTreeNode:
    __slots__ = {
        'name'               : None,
        'value_subnode_pairs': None,
    }
    
    def __init__(self, name, value_subnode_pairs):
        self.name = name
        self.value_subnode_pairs = value_subnode_pairs
    
    def __str__(self):
        return '{}:{{ name=\'{}\' }}'.format(object.__str__(self), self.name)
    
    def get_subnode_for_input(self, input):
        for value, subnode in self.value_subnode_pairs:
            if (input == value): return subnode
        return None

def id3(exemplars):
    '''Creates a decision tree from the exemplars using the ID3 algorithm.
    Each exemplar should be a 2-tuple of the format (inputs, output).
    The inputs of each exemplar should be a dict where each key is an
    attribute name and each key's value is the attribute value.
    
    Example usage:
    >>> from id3 import id3
    >>> # Exemplars for a simple 2-input logical OR with inputs x1 and x2
    >>> exemplars = (
    >>>     ({'x1': 0, 'x2': 0}, 0),
    >>>     ({'x1': 0, 'x2': 1}, 1),
    >>>     ({'x1': 1, 'x2': 0}, 1),
    >>>     ({'x1': 1, 'x2': 1}, 1),
    >>> )
    >>> id3_tree = id3(exemplars)
    >>> id3_tree.validate(exemplars)    # Make sure tree is valid
    True
    >>> id3_tree({'x1': 0, 'x2': 0})    # 0 or 0 is 0
    0
    >>> id3_tree({'x1': 0, 'x2': 1})    # 0 or 1 is 1
    1
    >>> id3_tree({'x1': 0, 'x2': 2})    # Unknown attribute value gives None
    None
    >>> 
    '''
    
    def id3_helper(exemplars):
        # Get the set of possible output values
        output_set = {output for _, output in exemplars}
        # Return None if there are no exemplars
        if   (len(output_set) == 0): return None
        # Return the output value if there is only one output value
        elif (len(output_set) == 1): return exemplars[0][1]
        
        # Determine if each input has only a single value across all exemplars
        inputs_all_same = True
        for inputs0, _ in exemplars:
            for inputs1, _ in exemplars:
                if (inputs0 != inputs1):
                    inputs_all_same = False
                    break
            if not inputs_all_same: break
        # All inputs are the same?
        if inputs_all_same:
            # Choose the most common output
            outputs = [output for _, output in exemplars]
            most_common       = None
            most_common_count = 0
            for output in set(outputs):
                count = outputs.count(output)
                if (count > most_common_count):
                    most_common       = output
                    most_common_count = count
            return most_common
        
        # Find the best attribute (lowest entropy) to partition the exemplars
        best_attr = None
        for attr in exemplars[0][0].keys():
            # Get the set of possible attribute values for this attribute
            attr_values = {inputs[attr] for inputs, _ in exemplars}
            
            # Partition the current attribute based on its possible values.
            # The keys in the partitions dict represent a possible attribute
            # value, and the value associated with each key is a list of
            # exemplars belonging to that attribute value.
            partitions = {}
            for attr_value in attr_values: partitions[attr_value] = []
            for inputs, output in exemplars:
                inputs     = dict(inputs)
                attr_value = inputs.pop(attr)
                partitions[attr_value].append((inputs, output))
            
            # Calculate the attribute entropy
            attr_entropy = 0
            for attr_value in partitions:
                outputs = [output for _, output in partitions[attr_value]]
                for output in set(outputs):
                    output_count  = outputs.count(output)
                    attr_entropy -= output_count*log2(output_count/len(outputs))
            
            # Determine if the current attribute is better than the current best
            if not best_attr or (attr_entropy < best_attr['entropy']):
                best_attr = {
                    'attr'      : attr,
                    'entropy'   : attr_entropy,
                    'partitions': partitions,
                }
        
        # Create and return the decision tree node for the best attribute
        value_subnode_pairs = []
        for attr_value in best_attr['partitions']:
            partition = best_attr['partitions'][attr_value]
            subnode   = id3_helper(partition)
            value_subnode_pairs.append((attr_value, subnode))
        
        return ID3DecisionTreeNode(best_attr['attr'], value_subnode_pairs)
    
    return ID3DecisionTree(id3_helper(exemplars))
