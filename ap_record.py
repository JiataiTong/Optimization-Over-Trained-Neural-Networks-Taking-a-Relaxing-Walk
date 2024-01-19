
import numpy as np


class RecordNode:
    def __init__(self, isLeaf):
        self.isLeaf = isLeaf
        self.left_child = None
        self.right_child = None
        self.val = - float('inf')


class RecordTree:
    def __init__(self, depth):
        self.depth = depth
        self.record_num = 0
        self.rootNode = RecordNode(False)

    def check_and_record(self, activation_pattern):
        flatten_pattern = np.hstack(activation_pattern)
        ifRecord = True
        current = self.rootNode
        n = 1
        for z in flatten_pattern:
            # print(z)
            if abs(z) <= 1e-6:
                n += 1
                if current.left_child is None and current.isLeaf is False:
                    ifRecord = False
                    isLeaf = False
                    if n == self.depth:
                        isLeaf = True
                    current.left_child = RecordNode(isLeaf)
                if not current.isLeaf:
                    current = current.left_child
            else:
                n += 1
                if current.right_child is None and current.isLeaf is False:
                    ifRecord = False
                    isLeaf = False
                    if n == self.depth:
                        isLeaf = True
                    current.right_child = RecordNode(isLeaf)
                if not current.isLeaf:
                    current = current.right_child
        if not ifRecord:
            self.record_num += 1

        return ifRecord

