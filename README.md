# python3-id3
Python3 implementation of the ID3 decision tree algorithm.

Example usage:
```Python
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
```
