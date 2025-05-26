from torch.nn import Module
import networkx as nx


class EnhancedGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph

    def visit_module(
        self, module: Module, module_name: str, parent_name: str = ""
    ) -> None:
        """Adds a module as a node and creates an edge to the parent."""
        # Add the module as a node with attributes
        self.graph.add_node(
            module_name,
            type=type(module).__name__,
            params=sum(p.numel() for p in module.parameters() if p.requires_grad),
        )
        # Add an edge from parent to current node if parent exists
        if parent_name:
            self.graph.add_edge(parent_name, module_name)

    def visit(self, module: Module, parent_name: str = "") -> None:
        """Recursively visit the module hierarchy."""
        for name, child in module.named_children():
            full_name = f"{parent_name}/{name}" if parent_name else name
            self.visit_module(child, full_name, parent_name)
            self.visit(child, full_name)


class ForwardGraphBuilder(EnhancedGraphBuilder):
    def __init__(self):
        super().__init__()
        self.forward_edges = []

    def register_hooks(self, model):
        """Registers hooks on all modules to capture forward connections."""
        for name, module in model.named_modules():
            module.register_forward_hook(self.forward_hook(name))

    def forward_hook(self, module_name):
        """Captures connections dynamically during forward pass."""

        def hook(module, input, output):
            for inp in input:
                if isinstance(inp, torch.Tensor):
                    self.graph.add_edge(module_name, str(inp.shape))
            if isinstance(output, torch.Tensor):
                self.graph.add_edge(module_name, str(output.shape))

        return hook


import matplotlib.pyplot as plt


def visualize_graph(graph, show_labels=True, file_name=None):
    """Visualize the graph with node and edge labels."""
    pos = nx.spring_layout(graph)  # Node positions
    plt.figure(figsize=(12, 10))

    # Filter nodes for those with a 'type' attribute to avoid KeyErrors
    valid_nodes = {
        node: data for node, data in graph.nodes(data=True) if "type" in data
    }

    # Draw only valid nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=valid_nodes.keys(),
        node_color="skyblue",
        node_size=5000,
        alpha=0.8,
    )

    # Draw all edges
    nx.draw_networkx_edges(graph, pos, edge_color="gray", arrows=True)

    # Draw labels for valid nodes
    if show_labels:
        node_labels = {
            node: f"{data['type']} ({data.get('params', 0)} params)"
            for node, data in valid_nodes.items()
        }
        nx.draw_networkx_labels(
            graph, pos, labels=node_labels, font_size=8, font_color="black"
        )

    # Display edge labels (if needed)
    edge_labels = {(u, v): f"{graph[u][v].get('label', '')}" for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    plt.title("PyTorch Model Graph", fontsize=14)
    plt.axis("off")

    # Save to file if specified
    if file_name:
        plt.savefig(file_name, format="png")
    plt.show()

    import matplotlib.pyplot as plt


def visualize_graph(graph, show_labels=True, file_name=None):
    """Visualize the graph with circles and detailed node information."""
    pos = nx.spring_layout(graph, seed=42)  # Fix seed for consistent layout
    plt.figure(figsize=(12, 10))

    # Draw nodes as circles
    nx.draw_networkx_nodes(graph, pos, node_color="skyblue", node_size=5000, alpha=0.8)

    # Draw edges with arrows
    nx.draw_networkx_edges(
        graph, pos, edge_color="gray", arrows=True, arrowstyle="-|>", arrowsize=20
    )

    # Add node labels
    if show_labels:
        node_labels = {}
        for node, data in graph.nodes(data=True):
            if "type" in data:
                layer_type = data["type"]
                if layer_type == "Linear":
                    in_features = data.get("in_features", "?")
                    out_features = data.get("out_features", "?")
                    params = data.get("params", 0)
                    label = (
                        f"{layer_type}\n{in_features} → {out_features}\n{params} params"
                    )
                elif layer_type == "Conv2d":
                    in_channels = data.get("in_channels", "?")
                    out_channels = data.get("out_channels", "?")
                    kernel_size = data.get("kernel_size", "?")
                    params = data.get("params", 0)
                    label = f"{layer_type}\n{in_channels} → {out_channels}\n{kernel_size}×{kernel_size} kernel\n{params} params"
                else:
                    params = data.get("params", 0)
                    label = f"{layer_type}\n{params} params"
                node_labels[node] = label
            else:
                node_labels[node] = node  # Fallback for nodes without attributes

        nx.draw_networkx_labels(
            graph, pos, labels=node_labels, font_size=8, font_color="black"
        )

    # Add edge labels (optional, if edges have labels)
    edge_labels = {(u, v): f"{graph[u][v].get('label', '')}" for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    plt.title("PyTorch Model Graph", fontsize=14)
    plt.axis("off")

    # Save to file if specified
    if file_name:
        plt.savefig(file_name, format="png", bbox_inches="tight")
    plt.show()


def export_graph(graph, file_name="model_graph"):
    """Export graph to DOT, JSON, and PNG formats."""
    nx.drawing.nx_pydot.write_dot(graph, f"{file_name}.dot")  # Export to DOT
    nx.write_gml(graph, f"{file_name}.gml")  # Export to GML
    nx.write_graphml(graph, f"{file_name}.graphml")  # Export to GraphML
    print(f"Graph exported to {file_name}.dot, .gml, .graphml")


import torch
import torch.nn as nn


# Example Model
class EnhancedModel(nn.Module):
    def __init__(self):
        super(EnhancedModel, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Instantiate the model and graph builder
model = EnhancedModel()
builder = ForwardGraphBuilder()

# Build the static graph
builder.visit(model)

# Register hooks for dynamic graph
builder.register_hooks(model)

# Run a forward pass to capture dynamic connections
dummy_input = torch.randn(1, 100)
model(dummy_input)

# Visualize the graph
visualize_graph(builder.graph, file_name="model_graph")

# Export the graph
export_graph(builder.graph)
