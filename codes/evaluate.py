import requests
import json
import numpy as np
from collections import OrderedDict

class Evaluate(object):
    def __init__(self):
        self.parts = OrderedDict([('f5nXa', 'TextPrepare'), 
                                  ('hTrz8', 'WordsTagsCount'), 
                                  ('0kUjR', 'BagOfWords'), 
                                  ('tLJV1', 'MultilabelClassification')])
        self.answers = {key: None for key in self.parts}

    @staticmethod  
    def ravel_output(output):
        '''
           If student accidentally submitted np.array with one
           element instead of number, this function will submit
           this number instead
        '''
        if isinstance(output, np.ndarray) and output.size == 1:
            output = output.item(0)
        return output
    def produce_part(self, part, output):
        self.answers[part] = output
        print("Current output for task {} is:\n {}".format(self.parts[part], output[:100] + '...'))

    def produce_tag(self, tag, output):
        part_id = [k for k, v in self.parts.items() if v == tag]
        if len(part_id) != 1:
            raise RuntimeError('cannot match tag with part_id: found {} matches'.format(len(part_id)))
        part_id = part_id[0]
        self.produce_part(part_id, str(self.ravel_output(output)))
