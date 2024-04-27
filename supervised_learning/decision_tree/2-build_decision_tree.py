#!/usr/bin/env python3
"""Builds a decision tree"""
import numpy as np


class Node:
    """The node Class"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Gets the maximum depth below the current node"""
        if self.is_leaf is True:
            return self.depth
        left_max_depth = 0
        right_max_depth = 0
        if self.left_child:
            left_max_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_max_depth = self.right_child.max_depth_below()
        return max(left_max_depth, right_max_depth)

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes below the current node"""
        counter = 0
        if only_leaves is False:
            counter += 1
        if self.is_leaf and only_leaves is True:
            counter += 0
        if self.left_child:
            counter += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            counter += self.right_child.count_nodes_below(only_leaves)
        return counter

    def left_child_add_prefix(self, text):
        """Adds the prefix of the left child"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Adds the prefix of the right child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0]
        for x in lines[1:]:
            new_text += "\n       " + x
        return (new_text)

    def __str__(self):
        """Prints the node and it's childs recursively"""
        if self.is_root:
            node_text = (
                f"root [feature={self.feature},"
                f"threshold={self.threshold}]")
        else:
            node_text = (
                f"-> node [feature={self.feature},"
                f" threshold={self.threshold}]")
        left_child_str = self.left_child_add_prefix(str(self.left_child))
        right_child_str = self.right_child_add_prefix(str(self.right_child))
        return f"{node_text}\n{left_child_str}{right_child_str}"


class Leaf(Node):
    """The leaf class"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth below the current node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Calculates the number of nodes below the current node"""
        return 1

    def __str__(self):
        return (f"-> leaf [value={self.value}] ")


class Decision_Tree():
    """The decision tree class"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Gets the maximum depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """"Gets the number of nodes in the decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return self.root.__str__()
