from typing import Any


class ModuleVisitor:
    def visit(self, module: Any, parent_name: str = "") -> None:
        """Visit a module and its children."""
        for name, child in module.named_children():
            full_name = f"{parent_name}/{name}" if parent_name else name
            self.visit_module(child, full_name)
            self.visit(child, full_name)

    def visit_module(self, module: Any, module_name: str) -> None:
        """Override to handle visiting a single module."""
        pass


from torch.nn import Module
import networkx as nx


class GraphBuilder(ModuleVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph

    def visit_module(self, module: Module, module_name: str) -> None:
        # Add the module as a node
        self.graph.add_node(module_name, type=type(module).__name__)
        print(f"Visited: {module_name} -> {type(module).__name__}")

    def visit(self, module: Module, parent_name: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{parent_name}/{name}" if parent_name else name
            # Add an edge to represent the connection
            if parent_name:
                self.graph.add_edge(parent_name, full_name)
            self.visit_module(child, full_name)
            self.visit(child, full_name)


import matplotlib.pyplot as plt


def visualize_graph(graph):
    pos = nx.spring_layout(graph)  # Node positions
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", font_weight="bold")
    plt.show()


import torch.nn as nn


# Example PyTorch Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Create the model
model = SimpleModel()

# Build and visualize the graph
builder = GraphBuilder()
builder.visit(model)
visualize_graph(builder.graph)
