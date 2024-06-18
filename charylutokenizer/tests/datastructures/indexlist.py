# A simple linked-list that maintains a (possibly stale) index to its pairs.
# The index maps from pair (X, Y) to all nodes that might contain X and whose
# next node might contain Y. The index is stale in the sense that we only add
# things to it, never remove. So when actually iterating on elements from the
# index, we need to make sure that the nodes still hold the desired pairs.
import sys


class IndexedList:
    class Node:
        __slots__ = "val", "prev", "next"

        def __init__(self, val, prev, next):
            self.val, self.prev, self.next = val, prev, next

        def delete(self):
            if self.prev is not None:
                self.prev.next = self.next
            if self.next is not None:
                self.next.prev = self.prev
            self.next = self.prev = None

        # def __sizeof__(self) -> int:
        #     return (
        #         sys.getsizeof(self.val)
        #         + sys.getsizeof(self.prev)
        #         + sys.getsizeof(self.next)
        #     )

    def __init__(self, l):
        self.c = 0
        self.stale_index = {}
        l = iter(l)
        a = next(l)
        self.start = prev_node = IndexedList.Node(a, None, None)
        for b in l:
            prev_node.next = node = IndexedList.Node(b, prev_node, None)
            self.index((a, b), prev_node)
            self.c += 1
            a, prev_node = b, node
        # no final adiciona um -1
        prev_node.next = node = IndexedList.Node(-1, prev_node, None)
        self.index((a, -1), prev_node)
        self.c += 1
        self.last = node

    def add(self, l):
        l = iter(l)
        a = next(l)
        prev_node = self.last
        for b in l:
            prev_node.next = node = IndexedList.Node(b, prev_node, None)
            self.index((a, b), prev_node)
            self.c += 1
            a, prev_node = b, node
        # no final adiciona um -1
        if prev_node.val != -1:
            prev_node.next = node = IndexedList.Node(-1, prev_node, None)
            self.index((a, -1), prev_node)
            self.c += 1
            self.last = node

    def __iter__(self):
        node = self.start
        while node is not None:
            yield node
            node = node.next

    def update_index(self, node):  # Update index before/after node.
        if node.prev is not None:
            self.index((node.prev.val, node.val), node.prev)
        if node.next is not None:
            self.index((node.val, node.next.val), node)

    def index(self, pair, node):
        self.stale_index.setdefault(pair, []).append(node)

    def count(self):
        # count = 0
        # for _ in self:
        #     count += 1
        return self.c

    def __sizeof__(self) -> int:
        total = sys.getsizeof(self.stale_index)
        for n in self:
            total += sys.getsizeof(n)
        return total
