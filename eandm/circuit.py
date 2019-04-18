import json 
import os 
import re 
import sys 

# Python 3 doesn't have xrange, and range behaves like xrange.
if sys.version_info >= (3,):
    xrange = range

"""
Circuit verification library.
"""

class Wire(object):
    """
    A wire in an on-chip circuit.
    Wires are immutable, and are either horizontal or vertical.
    """
  
    def __init__(self, name, x1, y1, x2, y2):
        """
        Create a wire. Raise a ValueError if the coordinates
        don't make up a horizontal wire or a vertical wire.
        name is the wire's user-visible name x1 is the X coordinate
        of the wire's first endpoint, y1 is the Y coordinate of the
        wire's first endpoint, x2 is the X coordinate of the wire's
        last endpoint, y2 isthe Y coordinate of the wire's last endpoint.
        """
        # Normalize the coordinates.
        if x1 > x2:
        x1, x2 = x2, x1
        if y1 > y2:
        y1, y2 = y2, y1

        self.name = name
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.object_id = Wire.next_object_id()

        if not (self.is_horizontal() or self.is_vertical()):
        raise ValueError(str(self) + " is neither horizontal nor vertical")
    
    def is_horizontal(self):
        """
        True if the wire"s endpoints have the same Y coordinates.
        """
        return self.y1 == self.y2

    def is_vertical(self):
        """
        True if the wire"s endpoints have the same X coordinates.
        """
        return self.x1 == self.x2
    
    def intersects(self, other_wire):
        """
        True if this wire intersects another wire.
        """
        # NOTE: we assume that wires can only cross, but not overlap.
        if self.is_horizontal() == other_wire.is_horizontal():
        return False

        if self.is_horizontal():
        h = self
        v = other_wire
        else:
        h = other_wire
        v = self
        return v.y1 <= h.y1 and h.y1 <= v.y2 and h.x1 <= v.x1 and v.x1 <= h.x2
    
    def __repr__(self):
        # :nodoc: nicer formatting to help with debugging
        return("<wire " + self.name + " (" + str(self.x1) + "," + str(self.y1) +
             ")-(" + str(self.x2) + "," + str(self.y2) + ")>")
    
    def as_json(self):
        """
        Dict that obeys the JSON format restrictions, representing the wire.
        """
        return {"id": self.name, "x": [self.x1, self.x2], "y": [self.y1, self.y2]}
  
    # Next number handed out by Wire.next_object_id()
    _next_id = 0
    
    @staticmethod
    def next_object_id():
        """Returns a unique numerical ID to be used as a Wire"s object_id."""
        id = Wire._next_id
        Wire._next_id += 1
        return id

class WireLayer(object):
    """
    The layout of one layer of wires in a chip.
    """

    def __init__(self):
        """
        Creates a layer layout with no wires.
        """
        self.wires = {}

    def wires(self):
        """
        The wires in the layout.
        """
        self.wires.values()

    def add_wire(self, name, x1, y1, x2, y2):
        """
        Adds a wire to a layer layout.
        name is the wire's unique name, x1 is the X coordinate of the wire"s first endpoint,
        y2 is the Y coordinate of the wire's first endpoint, x2 is the X coordinate of the
        wire's last endpoint, y2 is the Y coordinate of the wire's last endpoint.
        
        Raises an exception if the wire isn"t perfectly horizontal (y1 = y2) or
        perfectly vertical (x1 = x2).
        """
        if name in self.wires:
            raise ValueError("Wire name " + name + " not unique")
        self.wires[name] = Wire(name, x1, y1, x2, y2)
  
    def as_json(self):
        """
        Dict that obeys the JSON format restrictions, representing the layout.
        """
        return { "wires": [wire.as_json() for wire in self.wires.values()] }
  
  @staticmethod
    def from_file(file):
        """
        Builds a wire layer layout by reading a textual description from a file.
        file is a File object supplying the input
        Returns a new Simulation instance.
        """

        layer = WireLayer()
        
        while True:
            command = file.readline().split()
            if command[0] == "wire":
                coordinates = [float(token) for token in command[2:6]]
                layer.add_wire(command[1], *coordinates)
            elif command[0] == "done":
                break
          
        return layer

class RangeIndex(object):
    """
    Array-based range index implementation.
    """
  
    def __init__(self):
        """
        Initially empty range index.
        """
        self.data = []
  
    def add(self, key):
        """
        Inserts a key in the range index.
        """
        if key is None:
            raise ValueError("Cannot insert nil in the index")
        self.data.append(key)
  
    def remove(self, key):
        """
        Removes a key from the range index.
        """
        self.data.remove(key)
  
    def list(self, first_key, last_key):
        """
        List of values for the keys that fall within [first_key, last_key].
        """
        return [key for key in self.data if first_key <= key <= last_key]
  
    def count(self, first_key, last_key):
        """
        Number of keys that fall within [first_key, last_key].
        """
        result = 0
        for key in self.data:
            if first_key <= key <= last_key:
                result += 1
        return result

class BlitRangeIndex(object):
    """
    Sorted array-based range index implementation.
    """
  
    def __init__(self):
        """
        Initially empty range index.
        """
        self.data = []
  
    def add(self, key):
        """
        Inserts a key in the range index.
        """
        if key is None:
            raise ValueError("Cannot insert None in the index")
        self.data.insert(self._binary_search(key), key)
  
    def remove(self, key):
        """
        Removes a key from the range index.
        """
        index = self._binary_search(key)
        if index < len(self.data) and self.data[index] == key:
            self.data.pop(index)
  
    def list(self, low_key, high_key):
        """
        List of values for the keys that fall within [low_key, high_key].
        """
        low_index = self._binary_search(low_key)
        high_index = self._binary_search(high_key)
        return self.data[low_index:high_index]
  
    def count(self, low_key, high_key):
        """
        Number of keys that fall within [low_key, high_key].
        """
        low_index = self._binary_search(low_key)
        high_index = self._binary_search(high_key)
        if high_index < len(self.data) and self.data[high_index] == high_key:
            high_index += 1
        return high_index - low_index
  
    def _binary_search(self, key):
        """
        Binary search for the given key in the sorted array.
        """
        low, high = 0, len(self.data) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_key = self.data[mid]
            if key < mid_key:
                high = mid - 1
            elif key > mid_key:
                low = mid + 1
            else:
                return mid
        return high + 1

if os.environ.get("INDEX") == "blit":
    RangeIndex = BlitRangeIndex

class AvlRangeIndex(object):
    """
    Sorted array-based range index implementation.
    """
  
    def __init__(self):
        """
        Initially empty range index.
        """
    self.tree = RangeTree()
  
    def add(self, key):
        """
        Insert a key in the range index.
        """
    if key is None:
        raise ValueError("Cannot insert None in the index")
    self.tree.insert(key)
  
    def remove(self, key):
        """
        Remove a key from the range index.
        """
        self.tree.delete(key)
  
    def list(self, low_key, high_key):
        """
        List of values for the keys that fall within [low_key, high_key].
        """
        return [node.key for node in self.tree.list(low_key, high_key)]
  
    def count(self, low_key, high_key):
        """
        Number of keys that fall within [low_key, high_key].
        """
        return self.tree.rank(high_key) - self.tree.rank(low_key)

if os.environ.get("INDEX") == "avl":
    RangeIndex = AvlRangeIndex

class BSTNode(object):
  """
  A node in a vanilla BST tree.
  """
  
    def __init__(self, key):
        """
        Create a node that will be inserted in a BST.
        After the node is created, it should be inserted in a tree by calling
        BST.insert(). Until that happens, the node is not in a valid state.
        key is the key associated with the node
        """
        self.key = key
        self.parent = None
        self.left = None
        self.right = None
      
    def find(self, key):
        """
        The node with the given key in the subtree rooted at this node.
        key is the key of the node to be returned
        Return a BSTNode instance with the given key, or None if the key was not
        found.
        """
        if key < self.key:
            return self.left and self.left.find(key)
        elif key > self.key:
            return self.right and self.right.find(key)
        return self
  
    def min(self):
        """
        The node with the minimum key in the subtree rooted at this node.
        Return a BSTNode instance with the minimum key.
        """
        if self.left is None:
            return self
        return self.left.min()
     
    def successor(self):
        """
        The node with the next larger key (the successor) in the BST.
        Return a BSTNode instance, or None if this node has no successor.
        """
        if self.right is not None:
            return self.right.min()
        current = self
        while current.parent is not None and current is current.parent.right:
            current = current.parent
        return current.parent

    def insert(self, node):
        """
        Inserts a node into the subtree rooted at this node.
        node is the node to be inserted
        Return the node argument, if the node was inserted in the tree. If a node
        with the same key was found, that node is returned instead.
        """
        if node.key < self.key:
            if self.left is not None:
                return self.left.insert(node)
            node.parent = self
            self.left = node
            return node
        elif node.key > self.key:
            if self.right is not None:
                return self.right.insert(node)
            node.parent = self
            self.right = node
            return node
        return self

    def delete(self):
        """
        Delete this node from the BST.
        Return the deleted BSTNode instance. The instance might be different from
        this node, but will have this node"s key. The deleted node"s fields will
        still be set, despite the fact that it does not belong to the tree anymore.
        """
        if self.left is None or self.right is None:
            if self is self.parent.left:
                self.parent.left = self.left or self.right
            if self.parent.left is not None:
                self.parent.left.parent = self.parent
        else:
            self.parent.right = self.left or self.right
            if self.parent.right is not None:
                self.parent.right.parent = self.parent
            return self
        else:
            s = self.successor()
            # NOTE: deleting before swapping the keys so the BST RI is never violated.
            deleted_node = s.delete()
            self.key, s.key = s.key, self.key
        return deleted_node
  
    def check_ri(self):
        """
        Checks the BST representation invariant around this node.
        Raises an exception if the RI is violated.
        """
        if self.left is not None:
            if self.left.key >= self.key:
                raise RuntimeError("BST RI violated by a left node key")
            if self.left.parent is not self:
                raise RuntimeError("BST RI violated by a left node parent pointer")
            self.left.check_ri()
        if self.right is not None:
            if self.right.key <= self.key:
                raise RuntimeError("BST RI violated by a right node key")
            if self.right.parent is not self:
                raise RuntimeError("BST RI violated by a right node parent pointer")
            self.right.check_ri()

class BST(object):
    """
    A binary search tree.
    """
  
    def __init__(self, node_class = BSTNode):
        """
        Creates an empty BST.
       node_class (optional) is the class of nodes in the tree, defaults to BSTNode
        """
        self.node_class = node_class
        self.root = None

    def find(self, key):
        """
        The node with the given key in this BST.
        key is the key of the node to be returned
        Return a BSTNode instance with the given key, or None if the key was not
        found.
        """
        return self.root and self.root.find(key)
              
    def min(self):
        """
        The node with the minimum key in this BST.
        """
        if self.root is None:
            return None
        else:
            return self.root.min()
  
    def insert(self, key):
        """
        Insert a node into the subtree rooted at this node.
        key is the key of the node to be inserted
        Return a BSTNode with the given key that belongs to this tree.
        """
        node = self.node_class(key)
        if self.root is None:
            self.root = node
            return node
        return self.root.insert(node)
          
    def delete(self, key):
        """
        Remove the node with the given key from this BST.
        key is the key of the node to be deleted
        Return a BSTNode instance with the given key, which was removed from the
        tree. If this tree does not contain the given key, returns None. The deleted
        node's fields will still be set, despite the fact that it does not belong to
        the tree anymore.
        """
        node = self.find(key)
        if node is None:
            return None
        if node is self.root:
            pseudo_root = self.node_class(None)
            pseudo_root.left = self.root
            self.root.parent = pseudo_root
            deleted_node = self.root.delete()
            self.root = pseudo_root.left
            if self.root is not None:
                self.root.parent = None
            return deleted_node
        else:
            return node.delete()
      
    def successor(self, key):
        """
        Return the node that contains the next larger (the successor) key in
        the BST in relation to the node with key k.
        key is the key of the node whose successor will be returned
        Return a BSTNode instance whose key is the successor of the given key, or
        None if the given key doesn"t exist in the tree, or if is the maximum key,
        so the corresponding node has no successor.
        """
        node = self.find(key)
        return node and node.successor()
  
    def check_ri(self):
        """
        Check the BST representation invariant.
        Raise an exception if the RI is violated.
        """
        if self.root is not None:
            if self.root.parent is not None:
                raise RuntimeError("BST RI violated by the root node"s parent pointer")
            self.root.check_ri()
    
class AVLNode(BSTNode):
    """
    A node in an AVL.
    """
  
    def __init__(self, key):
        """
        Create a node that will be inserted in an AVL.
        See __init__ in BSTNode.
        """
        BSTNode.__init__(self, key)
        self.height = 0

    def update_subtree_info(self):
        """
        Update pre-computed fields such as the node"s subtree height.
        """
        self.height = self._uncached_height()

    def _uncached_height(self):
        """
        Re-compute the node"s height based on the children"s heights.
        """
        return 1 + max((self.left and self.left.height) or -1,
                       (self.right and self.right.height) or -1)
    
  def check_ri(self):
        """
        Check the AVL representation invariant around this node.
        Raise an exception if the RI is violated.
        """
        if self.height != self._uncached_height():
            raise RuntimeError("AVL RI violated by node height")
        if abs(AVL._height(self.left) - AVL._height(self.right)) > 1:
            raise RuntimeError("AVL RI violated by unbalanced children heights")
        BSTNode.check_ri(self)
    
class AVL(BST):
  """
  AVL binary search tree implementation.
  All queries and updates are guaranteed to take O(lg N) time.
  """
    def __init__(self, node_class = AVLNode):
        """
        Create an empty AVL tree.
        node_class (optional) is the class of nodes in the tree, defaults to AVLNode
        """
        BST.__init__(self, node_class)
  
    def insert(self, key):
        """
        Insert a node into the subtree rooted at this node.
        key is the key of the node to be inserted
        Return an AVLNode with the given key that belongs to this tree.
        """
        inserted_node = BST.insert(self, key)
        self._rebalance(inserted_node)
        return inserted_node

    def delete(self, key):
        """
        Remove the node with the given key from this BST.
        key is the key of the node to be deleted
        Return a BSTNode instance with the given key, which was removed from the
        tree. If this tree does not contain the given key, returns None. The deleted
        node"s fields will still be set, despite the fact that it does not belong to
        the tree anymore.
        """
        deleted_node = BST.delete(self, key)
        # NOTE: deleted_node still has its parent set, and it happens to be the
        #       first potentially out-of-balance node.
        self._rebalance(deleted_node.parent)
        return deleted_node
  
    def _rebalance(self, node):
        while node is not None:
          # NOTE: rebalance is called after an insertion or a deletion; asides from
          #       fixing the imbalance, it is also responsible for updating the
          #       cached sub-tree information in each node on the path to the root
      node.update_subtree_info()
      
        if AVL._height(node.left) >= 2 + AVL._height(node.right):
            if AVL._height(node.left.left) < AVL._height(node.left.right):
                self._left_rotate(node.left)
            self._right_rotate(node)
        elif AVL._height(node.right) >= 2 + AVL._height(node.left):
            if AVL._height(node.right.right) < AVL._height(node.right.left):
                self._right_rotate(node.right)
            self._left_rotate(node)
        node = node.parent

    @staticmethod
    def _height(node):
        if node is not None:
            return node.height
        else:
            return -1

    def _left_rotate(self, x):
        y = x.right
        y.parent = x.parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            else:  # y.parent.right is x
                y.parent.right = y
        x.right = y.left
        if x.right is not None:
            x.right.parent = x
        y.left = x
        x.parent = y
        x.update_subtree_info()
        y.update_subtree_info()

    def _right_rotate(self, x):
        y = x.left
        y.parent = x.parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            else:  # y.parent.right is x
                y.parent.right = y
        x.left = y.right
        if x.left is not None:
            x.left.parent = x
        y.right = x
        x.parent = y
        x.update_subtree_info()
        y.update_subtree_info()

class RangeNode(AVLNode):
    """
    A node in a range tree.
    """

    def __init__(self, key):
        """
        Create a node that will be inserted in a range index tree.
        See __init__ in BSTNode.
        """
        AVLNode.__init__(self, key)
        self.tree_size = 1

    def update_subtree_info(self):
        """
        Update pre-computed fields such as the node's subtree height.
        """
        AVLNode.update_subtree_info(self)
        self.tree_size = self._uncached_tree_size()

    def _uncached_tree_size(self):
        """
        Re-compute the node's subtree size based on the children's heights.
        """
        return 1 + (((self.left and self.left.tree_size) or 0) +
                ((self.right and self.right.tree_size) or 0))
    
    def check_ri(self):
        """
        Checks the AVL representation invariant around this node.
        Raises an exception if the RI is violated.
        """
        if self.tree_size != self._uncached_tree_size():
            raise RuntimeError("RangeTree RI violated by node subtree size")
        AVLNode.check_ri(self)

    def rank(self, key):
        """
        Number of keys <= the given key in the subtree rooted at this node.
        """
        if key < self.key:
            if self.left is not None:
                return self.left.rank(key)
            else:
                return 0
        if self.left:
            lrank = 1 + self.left.tree_size
        else:
            lrank = 1
        if key > self.key and self.right is not None:
            return lrank + self.right.rank(key)
        return lrank

    def lca(self, low_key, high_key):
        """
        Lowest-common ancestor node of nodes with low_key and high_key.
        If low_key and/or high_key are not in the tree, this returns the LCA of the
        nodes that would be created by inserting the keys in the tree.
        Returns a RangeNode instance, or None if low_key and high_key are not in the
        node"s subtree, and there is no key in the tree such that
        low_key < key < high_key.
        """
        if low_key <= self.key <= high_key:
            return self
        if low_key < self.key:
            return self.left and self.left.lca(low_key, high_key)
        else:
            return self.right and self.right.lca(low_key, high_key)

  def list(self, low_key, high_key, result):
    """
    Lists nodes with keys between low_key and high_key in this node's subtree.
    
    Extends result with a list of RangeNode instances for the nodes in the
    subtree rooted at this node, such that each node"s key is between low_key
    and high_key."""
    if low_key <= self.key <= high_key:
      result.append(self)
    if self.left is not None and low_key <= self.key:
      self.left.list(low_key, high_key, result)
    if self.right is not None and self.key <= high_key:
      self.right.list(low_key, high_key, result)

class RangeTree(AVL):
  """BST with support for range queries."""

  def __init__(self, node_class = RangeNode):
    """Creates an empty AVL tree.
    
    Args:
      node_class (optional): the class of nodes in the tree, defaults to AVLNode
    """
    AVL.__init__(self, node_class)
    
  def rank(self, key):
    """Number of keys <= the given key in the tree."""
    if self.root is not None:
      return self.root.rank(key)
    return 0

  def lca(self, low_key, high_key):
    """Lowest-common ancestor node of nodes with low_key and high_key.
    
    If low_key and/or high_key are not in the tree, this returns the LCA of the
    nodes that would be created by inserting the keys in the tree.
    
    Returns a RangeNode instance, or None if low_key and high_key are not in the
    tree, and there is no key in the tree such that low_key < key < high_key.
    """
    return self.root and self.root.lca(low_key, high_key)

  def list(self, low_key, high_key):
    """A list containing the nodes with keys between low_key and high_key."""
    result = []
    lca = self.lca(low_key, high_key)
    if lca is not None:
      lca.list(low_key, high_key, result)
    return result

  # Un-comment the lines below for RI checking.
  """
  def insert(self, key):
    result = AVL.insert(self, key)
    self.check_ri()
    return result
  
  def delete(self, key):
    result = AVL.delete(self, key)
    self.check_ri()
    return result
  """
  
class TracedRangeIndex(RangeIndex):
  """Augments RangeIndex to build a trace for the visualizer."""
  
  def __init__(self, trace):
    """Sets the object receiving tracing info."""
    RangeIndex.__init__(self)
    self.trace = trace
  
  def add(self, key):
    self.trace.append({"type": "add", "id": key.wire.name})
    RangeIndex.add(self, key)
  
  def remove(self, key):
    self.trace.append({"type": "delete", "id": key.wire.name})
    RangeIndex.remove(self, key)
  
  def list(self, first_key, last_key):
    result = RangeIndex.list(self, first_key, last_key)
    self.trace.append({"type": "list", "from": first_key.key,
                       "to": last_key.key,
                       "ids": [key.wire.name for key in result]})
    return result
  
  def count(self, first_key, last_key):
    result = RangeIndex.count(self, first_key, last_key)
    self.trace.append({"type": "list", "from": first_key.key,
                       "to": last_key.key, "count": result})
    return result

class ResultSet(object):
  """Records the result of the circuit verifier (pairs of crossing wires)."""
  
  def __init__(self):
    """Creates an empty result set."""
    self.crossings = []
  
  def add_crossing(self, wire1, wire2):
    """Records the fact that two wires are crossing."""
    self.crossings.append(sorted([wire1.name, wire2.name]))
  
  def write_to_file(self, file):
    """Write the result to a file."""
    for crossing in self.crossings:
      file.write(" ".join(crossing))
      file.write("\n")

class TracedResultSet(ResultSet):
  """Augments ResultSet to build a trace for the visualizer."""
  
  def __init__(self, trace):
    """Sets the object receiving tracing info."""
    ResultSet.__init__(self)
    self.trace = trace
    
  def add_crossing(self, wire1, wire2):
    self.trace.append({"type": "crossing", "id1": wire1.name,
                       "id2": wire2.name})
    ResultSet.add_crossing(self, wire1, wire2)

class KeyWirePair(object):
  """Wraps a wire and the key representing it in the range index.
  
  Once created, a key-wire pair is immutable."""
  
  def __init__(self, key, wire):
    """Creates a new key for insertion in the range index."""
    self.key = key
    if wire is None:
      raise ValueError("Use KeyWirePairL or KeyWirePairH for queries")
    self.wire = wire
    self.wire_id = wire.object_id

  def __lt__(self, other):
    # :nodoc: Delegate comparison to keys.
    return (self.key < other.key or
            (self.key == other.key and self.wire_id < other.wire_id))
  
  def __le__(self, other):
    # :nodoc: Delegate comparison to keys.
    return (self.key < other.key or
            (self.key == other.key and self.wire_id <= other.wire_id))  

  def __gt__(self, other):
    # :nodoc: Delegate comparison to keys.
    return (self.key > other.key or
            (self.key == other.key and self.wire_id > other.wire_id))
  
  def __ge__(self, other):
    # :nodoc: Delegate comparison to keys.
    return (self.key > other.key or
            (self.key == other.key and self.wire_id >= other.wire_id))

  def __eq__(self, other):
    # :nodoc: Delegate comparison to keys.
    return self.key == other.key and self.wire_id == other.wire_id
  
  def __ne__(self, other):
    # :nodoc: Delegate comparison to keys.
    return self.key != other.key or self.wire_id != other.wire_id

  def __hash__(self):
    # :nodoc: Delegate comparison to keys.
    return hash([self.key, self.wire_id])

  def __repr__(self):
    # :nodoc: nicer formatting to help with debugging
    return "<key: " + str(self.key) + " wire: " + str(self.wire) + ">"

class KeyWirePairL(KeyWirePair):
  """A KeyWirePair that is used as the low end of a range query.
  
  This KeyWirePair is smaller than all other KeyWirePairs with the same key."""
  def __init__(self, key):
    self.key = key
    self.wire = None
    self.wire_id = -1000000000

class KeyWirePairH(KeyWirePair):
  """A KeyWirePair that is used as the high end of a range query.
  
  This KeyWirePair is larger than all other KeyWirePairs with the same key."""
  def __init__(self, key):
    self.key = key
    self.wire = None
    # HACK(pwnall): assuming 1 billion objects won"t fit into RAM.
    self.wire_id = 1000000000

class CrossVerifier(object):
  """Checks whether a wire network has any crossing wires."""
  
  def __init__(self, layer):
    """Verifier for a layer of wires.
    
    Once created, the verifier can list the crossings between wires (the 
    wire_crossings method) or count the crossings (count_crossings)."""

    self.events = []
    self._events_from_layer(layer)
    self.events.sort()
  
    self.index = RangeIndex()
    self.result_set = ResultSet()
    self.performed = False
  
  def count_crossings(self):
    """Returns the number of pairs of wires that cross each other."""
    if self.performed:
      raise 
    self.performed = True
    return self._compute_crossings(True)

  def wire_crossings(self):
    """An array of pairs of wires that cross each other."""
    if self.performed:
      raise 
    self.performed = True
    return self._compute_crossings(False)

  def _events_from_layer(self, layer):
    """Populates the sweep line events from the wire layer."""
    left_edge = min([wire.x1 for wire in layer.wires.values()])
    for wire in layer.wires.values():
      if wire.is_horizontal():
        self.events.append([left_edge, 0, wire.object_id, "add", wire])
      else:
        self.events.append([wire.x1, 1, wire.object_id, "query", wire])

  def _compute_crossings(self, count_only):
    """Implements count_crossings and wire_crossings."""
    if count_only:
      result = 0
    else:
      result = self.result_set

    for event in self.events:
      event_x, event_type, wire = event[0], event[3], event[4]
      
      if event_type == "add":
        self.index.add(KeyWirePair(wire.y1, wire))
      elif event_type == "query":
        self.trace_sweep_line(event_x)
        cross_wires = []
        for kwp in self.index.list(KeyWirePairL(wire.y1),
                                   KeyWirePairH(wire.y2)):
          if wire.intersects(kwp.wire):
            cross_wires.append(kwp.wire)
        if count_only:
          result += len(cross_wires)
        else:
          for cross_wire in cross_wires:
            result.add_crossing(wire, cross_wire)

    return result
  
  def trace_sweep_line(self, x):
    """When tracing is enabled, adds info about where the sweep line is.
    
    Args:
      x: the coordinate of the vertical sweep line
    """
    # NOTE: this is overridden in TracedCrossVerifier
    pass
  
  if os.environ.get("CROSS") == "sweep":
    def _events_from_layer(self, layer):
      """Populates the sweep line events from the wire layer."""
      for wire in layer.wires.values():
        if wire.is_horizontal():
          self.events.append([wire.x1, 0, wire.object_id, "add", wire])
          self.events.append([wire.x2, 2, wire.object_id, "remove", wire])
        else:
          self.events.append([wire.x1, 1, wire.object_id, "query", wire])
  
    def _compute_crossings(self, count_only):
      """Implements count_crossings and wire_crossings."""
      if count_only:
        result = 0
      else:
        result = self.result_set
  
      for event in self.events:
        event_x, event_type, wire = event[0], event[3], event[4]
        self.trace_sweep_line(event_x)
        
        if event_type == "add":
          self.index.add(KeyWirePair(wire.y1, wire))
        elif event_type == "remove":
          self.index.remove(KeyWirePair(wire.y1, wire))
        elif event_type == "query":
          if count_only:
            result += self.index.count(KeyWirePairL(wire.y1),
                                       KeyWirePairH(wire.y2))
          else:
            for kwp in self.index.list(KeyWirePairL(wire.y1),
                                       KeyWirePairH(wire.y2)):
              result.add_crossing(wire, kwp.wire)
    
      return result
  
class TracedCrossVerifier(CrossVerifier):
  """Augments CrossVerifier to build a trace for the visualizer."""
  
  def __init__(self, layer):
    CrossVerifier.__init__(self, layer)
    self.trace = []
    self.index = TracedRangeIndex(self.trace)
    self.result_set = TracedResultSet(self.trace)
    
  def trace_sweep_line(self, x):
    self.trace.append({"type": "sweep", "x": x})
    
  def trace_as_json(self):
    """List that obeys the JSON format restrictions with the verifier trace."""
    return self.trace
    

class TruthTable:
    """Truth table representation of the logic inside a gate."""
    
    def __init__(self, name, output_list):
        """Creates a truth table from a list representation.
        
        Args:
            name: User-friendly name for the truth table.
            output_list: The entries in the truth table, in the standard order
                (the inputs should look like an incrementing counter).
        
        Raises:
            TypeError: An exception if the list"s length is not a power of two.
        """
        self.name = name
        self.table = self._build_table(output_list)
        self.input_count = self._table_depth(self.table)

    def output(self, inputs):
        """Computes the output for this truth table, given a list of inputs."""
        if len(inputs) != self.input_count:
            raise ValueError("Inputs list is incorrectly sized")
        value = self.table
        for i in inputs:
            value = value[i]
        return value

    def _build_table(self, output_list):
        # Builds an evaluation table out of a list of truth table values.
        #
        # Raises:
        #    TypeError: An exception if the list"s length is not a power of two.
        if len(output_list) == 2:
            for value in output_list:
                if value != 0 and value != 1:
                    raise TypeError("Invalid value in truth output list")
            return output_list
        else:
            length = len(output_list)
            if length % 2 != 0:
                raise ValueError("Invalid truth output list length")
            half = length // 2
            return [self._build_table(output_list[0:half]),
                    self._build_table(output_list[half:])]

    def _table_depth(self, table):
        # The depth (number of inputs) of a truth table.
        depth = 0
        while table != 0 and table != 1:
            depth += 1
            table = table[0]
        return depth
    
class GateType:
    """A type of gate, e.g. 2-input NAND with 60ps delay."""
    
    def __init__(self, name, truth_table, delay):
        """Creates a gate type with a truth table and output delay.
        
        Args:
            name: User-friendly name for the gate type.
            truth_table: TruthTable instance containing the gate"s logic.
            delay: The time it takes an input transition to cause an output
                transition.
        
        Raises:
            ValueError: An exception if the delay is negative.
        """
        self.name = name
        if delay < 0:
            raise ValueError("Invalid delay")
        self.truth_table = truth_table
        self.input_count = truth_table.input_count
        self.delay = delay

    def output(self, inputs):
        """The gate"s output value, given a list of inputs."""
        return self.truth_table.output(inputs)
    
    def output_time(self, input_time):
        """The time of the gate"s output transition.
        
        Computes the time of the output transition given an input transition
        time.
        
        Args:
            input_time: Time of the input transition.
        """
        return self.delay + input_time

class Gate:
    """A gate in a circuit."""

    def __init__(self, name, gate_type):
        """ Creates an unconnected gate whose initial output is false.
        
        Args:
            name: User-friendly name for the gate.
            gate_type: GateType instance specifying the gate"s behavior.
        """
        self.name = name
        self.gate_type = gate_type
        self.in_gates = [None for i in xrange(gate_type.input_count)]
        self.out_gates = []
        self.probed = False
        self.output = 0
  
    def connect_input(self, gate, terminal):
        """Connects one of this gate"s input terminals to another gate's output.
        
        Args:
            gate: The gate whose output terminal will be connected.
            terminal: The number of this gate"s input terminal that will be
                connected (using 0-based indexing)
        """
        if self.in_gates[terminal] is not None:
            raise RuntimeError("Input terminal already connected")
        self.in_gates[terminal] = gate
        gate.out_gates.append(self)
      
    def probe(self):
        """Marks this gate as probed.
        
        So the simulator will record its transitions.
        
        Raises:
            RuntimeError: An exception if the gate is already probed.
        """
        if self.probed:
            raise RuntimeError("Gate already probed")
        self.probed = True

    def has_inputs_connected(self):
        """True if all the gate"s input terminals are connected to other gates.
        """
        for input in self.in_gates:
            if input == None:
                return False
        return True
  
    def has_output_connected(self):
        """True if the gate"s output terminal is connected to another gate."""
        return self.out_gates.length > 0
  
    def is_connected(self):
        """True if all the gate"s inputs and outputs are connected."""
        return self.has_inputs_connected and self.has_output_connected

    def transition_output(self):
        """The value that the gate"s output will have after transition.
        
        The gate"s output will not reflect this value right away. Each gate has
        a delay from its inputs" transitions to the output"s transition. The
        circuit simulator is responsible for setting the appropriate time.
        """
        return self.gate_type.output([gate.output for gate in self.in_gates])
  
    def transition_time(self, input_time):
        """The time that the gate"s output will reflect a change in its inputs.
        
        Args:
            input_time: The last time when the gate"s inputs changed.
        """
        return self.gate_type.output_time(input_time)
    
    def as_json(self):
        """"A hash that obeys the JSON format, representing the gate."""
        return {"id": self.name, "table": self.gate_type.truth_table.name,
                "type": self.gate_type.name, "probed": self.probed,
                "inputs": [g and g.name for g in self.in_gates],
        "outputs": [g and g.name for g in self.out_gates]}

class Circuit:
    """The topology of a combinational circuit, and a snapshot of its state.
    
    This class contains topological information about a circuit (how the gates
    are connected to each other) as well as information about the gates" states
    (values at their output terminals) at an instance of time.
    """
    def __init__(self):
        """Creates an empty circuit."""
        self.truth_tables = {}
        self.gate_types = {}
        self.gates = {}

    def add_truth_table(self, name, output_list):
        """Adds a truth table that can be later attached to gate types.
        
        Args:
            name: A unique string used to identify the truth table.
            output_list: A list of outputs for the truth table.
        
        Returns:
            A newly created TruthTable instance.
        """
        if name in self.truth_tables:
            raise ValueError("Truth table name already used")
        self.truth_tables[name] = TruthTable(name, output_list)
    
    def add_gate_type(self, name, truth_table_name, delay):
        """Adds a gate type that can be later attached to gates.
        
        Args:
            name: A unique string used to identify the gate type.
            truth_table_name: The name of the gate"s truth table.
            delay: The gate"s delay from an input transition to an output
                transition.
        
        Returns:
            The newly created GateType instance.
        """
        if name in self.gate_types:
            raise ValueError("Gate type name already used")
        truth_table = self.truth_tables[truth_table_name]
        if delay < 0:
            raise ValueError("Invalid delay")
        self.gate_types[name] = GateType(name, truth_table, delay)
    
    def add_gate(self, name, type_name, input_names):
        """Adds a gate and connects it to other gates.
        
        Args:
            name: A unique string used to identify the gate.
            type_name: The name of the gate"s type.
            input_names: List of the names of gates whose outputs are connected
                to this gate"s inputs.
        
        Returns:
            The newly created Gate instance.
        """
        if name in self.gates:
            raise ValueError("Gate name already used")
        gate_type = self.gate_types[type_name]
        self.gates[name] = new_gate = Gate(name, gate_type)
        for i in xrange(len(input_names)):
            gate = self.gates[input_names[i]]
            new_gate.connect_input(gate, i)
        return new_gate
    
    def add_probe(self, gate_name):
        """Adds a gate to the list of outputs."""
        gate = self.gates[gate_name]
        gate.probe()
      
    def as_json(self):
        """A hash that obeys the JSON format, representing the circuit."""
        json = {}
        json["gates"] = [gate.as_json() for gate in self.gates.itervalues()]
        return json

class Transition:
    """A transition in a gate's output."""
  
    def __init__(self, gate, new_output, time):
        """Creates a potential transition of a gate's output to a new value.
        
        Args:
            gate: The Gate whose output might transition.
            new_output: The new output value that the gate will take.
            time: The time at which the Gate"s output will match the new value.
        
        Raises:
            ValueError: An exception if the output is not 0 or 1.
        """
        if new_output != 0 and new_output != 1:
            raise ValueError("Invalid output value")
        self.gate = gate
        self.new_output = new_output
        self.time = time
        self.object_id = Transition.next_object_id()
    
    def __lt__(self, other):
        # :nodoc: Transitions should be comparable.
        return (self.time < other.time or
                (self.time == other.time and self.object_id < other.object_id))
    
    def __le__(self, other):
        # :nodoc: Transitions should be comparable.
        return (self.time < other.time or
                (self.time == other.time and self.object_id <= other.object_id))
    
    def __gt__(self, other):
        # :nodoc: Transitions should be comparable.
        return (self.time > other.time or
                (self.time == other.time and self.object_id > other.object_id))
    
    def __ge__(self, other):
        # :nodoc: Transitions should be comparable.
        return (self.time > other.time or
                (self.time == other.time and self.object_id >= other.object_id))
    
    # NOTE: Due to the comparisons" use of object_id, no two Transitions will be
    #       equal. So we don"t need to override __eq__, __ne__, or __hash__.
      
    def is_valid(self):
        """True if the transition would cause an actual change in the gate"s
        output.
        """
        return self.gate.output != self.new_output
    
    def apply(self):
        """Makes this transition effective by changing the gate's output.
        
        Raises:
            ValueError: An exception if applying the transition wouldn"t cause
                an actual change in the gate"s output.
        """
        if self.gate.output == self.new_output:
            raise ValueError("Gate output should not transition to the same "
                             "value")
        self.gate.output = self.new_output
    
    def __repr__(self):
        # :nodoc: debug output
        return ("<Transition at t=" + str(self.time) + ", gate " +
                self.gate.name + " -> " + str(self.new_output) + ">")
    
    # Next number handed out by Transition.next_object_id()
    _next_id = 0
    
    @staticmethod
    def next_object_id():
        """Returns a unique numerical ID to be used as a Transition"s object_id.
        """
        id = Transition._next_id
        Transition._next_id += 1
        return id

class PriorityQueue:
    """Heap-based priority queue implementation."""
    
    def __init__(self):
        """Initially empty priority queue."""
        self.heap = [None]
    
    def __len__(self):
        # Number of elements in the queue.
        return len(self.heap) - 1
    
    def append(self, key):
        """Inserts an element in the priority queue."""
        if key is None:
            raise ValueError("Cannot insert None in the queue")
    
        i = len(self.heap)
        self.heap.append(key)
        while i > 1:
            parent = i // 2
            if key < self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], key
                i = parent
            else:
                break
    
    def min(self):
        """Returns the smallest element in the queue."""
        return self.heap[1]
    
    def pop(self):
        """Removes the minimum element in the queue.
    
        Returns:
            The value of the removed element.
        """
        heap = self.heap
        popped_key = heap[1]
        if len(heap) == 2:
            return heap.pop()
        heap[1] = key = heap.pop()
        
        i = 1
        while True:
            left = i * 2
            if len(heap) <= left:
                break
            left_key = heap[left]
            right = i * 2 + 1
            right_key = right < len(heap) and heap[right]
            if right_key and right_key < left_key:
                child_key = right_key
                child = right
            else:
                child_key = left_key
                child = left
            if key <= child_key:
                break
            self.heap[i], self.heap[child] = child_key, key
            i = child
        return popped_key


class Simulation:
    """State needed to compute a circuit's state as it evolves over time."""
    
    def __init__(self, circuit):
        """Creates a simulation that will run on a pre-built circuit.
        
        The Circuit instance does not need to be completely built before it is
        given to the class constructor. However, it does need to be complete
        before the run method is called.
        
        Args:
            circuit: The circuit whose state transitions will be simulated.
        """
        self.circuit = circuit
        self.in_transitions = []
        
        self.queue = PriorityQueue()
        self.probes = []
        self.probe_all_undo_log = []

    def add_transition(self, gate_name, output_value, output_time):
        """Adds a transition to the simulation"s initial conditions.
        
        The transition should involve one of the circuit"s input gates.
        """
        gate = self.circuit.gates[gate_name]
        self.in_transitions.append([output_time, gate_name, output_value, gate])
    
    def step(self):
        """Runs the simulation for one time slice.
        
        A step does not equal one unit of time. The simulation logic ignores
        time units where nothing happens, and bundles all the transitions that
        happen at the same time in a single step.
        
        Returns:
            The simulation time after the step occurred.
        """
        step_time = self.queue.min().time
        
        # Need to apply all the transitions at the same time before propagating.
        transitions = []
        while len(self.queue) > 0 and self.queue.min().time == step_time:
          transition = self.queue.pop()
          if not transition.is_valid():
            continue
          transition.apply()
          if transition.gate.probed:
            self.probes.append([transition.time, transition.gate.name,
                                transition.new_output])
          transitions.append(transition)
        
        # Propagate the transition effects.
        for transition in transitions:
          for gate in transition.gate.out_gates:
            output = gate.transition_output()
            time = gate.transition_time(step_time)
            self.queue.append(Transition(gate, output, time))
        
        return step_time
    
    def run(self):
        """Runs the simulation to completion."""
        for in_transition in sorted(self.in_transitions):
            self.queue.append(Transition(in_transition[3], in_transition[2],
                                         in_transition[0]))
        while len(self.queue) > 0:
            self.step()
        self.probes.sort()
            
    def probe_all_gates(self):
        """Turns on probing for all gates in the simulation."""
        for gate in self.circuit.gates.itervalues():
            if not gate.probed:
                self.probe_all_undo_log.append(gate)
                gate.probe()

    def undo_probe_all_gates(self):
        """Reverts the effects of calling probe_all_gates!"""
        for gate in self.probe_all_undo_log:
            gate.probed = False
        self.probe_all_undo_log = []
    
    @staticmethod
    def from_file(file):
        """Builds a simulation by reading a textual description from a file.
        
        Args:
            file: A File object supplying the input.
        
        Returns: A new Simulation instance.
        """
        circuit = Circuit()
        simulation = Simulation(circuit)
        
        while True:
            command = file.readline().split()
            if len(command) < 1:
                continue
            if command[0] == "table":
                outputs = [int(token) for token in command[2:]]
                circuit.add_truth_table(command[1], outputs)
            elif command[0] == "type":
                if len(command) != 4:
                    raise ValueError("Invalid number of arguments for gate type"
                                     " command")
                circuit.add_gate_type(command[1], command[2], int(command[3]))
            elif command[0] == "gate":
                circuit.add_gate(command[1], command[2], command[3:])
            elif command[0] == "probe":
                if len(command) != 2:
                    raise ValueError("Invalid number of arguments for gate "
                                      "probe command")
                circuit.add_probe(command[1])
            elif command[0] == "flip":
                if len(command) != 4:
                    raise ValueError("Invalid number of arguments for flip "
                                     "command")
                simulation.add_transition(command[1], int(command[2]),
                                          int(command[3]))
            elif command[0] == "done":
                break
        return simulation
    
    def layout_from_file(self, file):
        """Reads the simulation's visual layout from a file.
        
        Args:
            file: A File-like object supplying the input.
        
        Returns:
             self.
        """
        while True:
          line = file.readline()
          if len(line) == 0:
              raise ValueError("Input lacks circuit layout information")
          if line.strip() == "layout":
              svg = file.read()
              # Get rid of the XML doctype.
              svg = re.sub("\\<\\?xml.*\\?\\>", '", svg)
              svg = re.sub("\\<\\!DOCTYPE[^>]*\\>", '", svg)
              self.layout_svg = svg.strip()
              break
        self
    
    def trace_as_json(self):
        """A hash that obeys the JSON format, containing simulation data."""
        return {"circuit": self.circuit.as_json(), "trace": self.probes,
                "layout": self.layout_svg}
    
    def outputs_to_line_list(self):
        return [" ".join([str(probe[0]), probe[1], str(probe[2])]) for probe in self.probes]
    
    def outputs_to_file(self, file):
        """Writes a textual description of the simulation's probe results to a
        file.
        
        Args:
            file: A File object that receives the probe results.
        """
        for line in self.outputs_to_line_list():
            file.write(line)
            file.write("\n")
            
    def jsonp_to_file(self, file):
        """Writes a JSONP description of the simulation's probe results to a
        file.
        
        Args:
            file: A File object that receives the probe results.
        """
        file.write("onJsonp(")
        json.dump(self.trace_as_json(), file)
        file.write(");\n")

# Command-line controller.
if __name__ == "__main__":
    import sys
    layer = WireLayer.from_file(sys.stdin)
    verifier = CrossVerifier(layer)
    
    if os.environ.get("TRACE") == "jsonp":
      verifier = TracedCrossVerifier(layer)
      result = verifier.wire_crossings()
      json_obj = {"layer": layer.as_json(), "trace": verifier.trace_as_json()}
      sys.stdout.write("onJsonp(")
      json.dump(json_obj, sys.stdout)
      sys.stdout.write(");\n")
    elif os.environ.get("TRACE") == "list":
      verifier.wire_crossings().write_to_file(sys.stdout)
    else:
      sys.stdout.write(str(verifier.count_crossings()) + "\n")
