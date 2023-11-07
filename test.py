class Node:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

class Tree:
    def __init__(self, root_data):
        self.root = Node(root_data)

    def create_children(self):
        child1 = Node("Child 1")
        child2 = Node("Child 2")
        child3 = Node("Child 3")

        self.root.add_child(child1)
        self.root.add_child(child2)
        self.root.add_child(child3)

    def display_tree(self, node=None, level=0):
        if node is None:
            node = self.root

        print("  " * level + node.data)

        for child in node.children:
            self.display_tree(child, level + 1)

# Usage example:
if __name__ == "__main__":
    my_tree = Tree("Root")
    my_tree.create_children()
    my_tree.display_tree()