#!/usr/bin/env python3
import numpy as np


class Node:
    """Represents a node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node object.

        Args:
            feature (int or None): The index of the feature
                                    used for splitting.
            threshold (float or None): The threshold value
                                       for splitting the data.
            left_child (Node or None): The left child node.
            right_child (Node or None): The right child node.
            is_root (bool): Indicates if the node is the root
                                      of the tree.
            depth (int): The depth of the node in the tree.
        """

        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Returns the maximum depth below the current node.

        Returns:
            int: The maximum depth below the current node.
        """

        if self.is_leaf:
            return self.depth
        left_max_depth = 0
        right_max_depth = 0
        if self.left_child:
            left_max_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_max_depth = self.right_child.max_depth_below()
        return max(left_max_depth, right_max_depth)


class Leaf(Node):
    """Represents a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """Initializes a Leaf object.

        Args:
            value: The predicted value of the leaf node.
            depth (int or None): The depth of the leaf node in the tree.
        """

        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf node.

        Returns:
            int: The depth of the leaf node.
        """

        return self.depth


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a Decision_Tree object.

        Args:
            max_depth (int): The maximum depth of the decision tree.
            min_pop (int): The minimum number of samples required
                           to split a node.
            seed (int): The seed value for random number generation.
            split_criterion (str): The criterion used for splitting nodes.
            root (Node or None): The root node of the decision tree.
        """

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
        """Returns the maximum depth of the entire tree.

        Returns:
            int: The maximum depth of the entire tree.
        """

        return self.root.max_depth_below()
