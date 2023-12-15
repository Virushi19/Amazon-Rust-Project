use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

// Directed Graph Struct
struct Graph {
    edges: HashMap<usize, Vec<usize>>,
}
impl Graph {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }
    fn add_edge(&mut self, from: usize, to: usize) {
        self.edges
            .entry(from)
            .or_insert_with(Vec::new)
            .push(to);
        self.edges.entry(to).or_insert_with(Vec::new);
    }
    // General Stats:
    // Graph Density
    fn graph_density(&self) -> f64 {
        let num_nodes = self.edges.len() as f64;
        let num_edges: f64 = self.edges.values().map(|v| v.len() as f64).sum();
        if num_nodes <= 1.0 {
            0.0
        } else {
            num_edges / (num_nodes * (num_nodes - 1.0))
        }
    }
    // Average Degree
    fn average_degree(&self) -> f64 {
        let num_nodes = self.edges.len() as f64;
        let num_edges: f64 = self.edges.values().map(|v| v.len() as f64).sum();
        if num_nodes == 0.0 {
            0.0
        } else {
            num_edges / num_nodes
        }
    }
    // Analysis Methods:
    // Degree Distribution of the graph
    fn degree_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (_, neighbors) in &self.edges {
            let degree = neighbors.len();
            *distribution.entry(degree).or_insert(0) += 1;
        }
        distribution
    }
    // Graph Clustering Coefficient
    fn graph_clustering(&self) -> (usize, f64) {
        let mut node_clusters: Vec<_> = self.edges.keys().collect();
        node_clusters.sort();

        let mut total_edges = 0;
        let cluster_edges: Vec<_> = node_clusters
            .iter()
            .map(|&node| {
                let edges = self.edges.get(node).unwrap().len();
                total_edges += edges;
                edges
            })
            .collect();
        let avg_degree = total_edges as f64 / node_clusters.len() as f64;
        let var_degree = cluster_edges
            .iter()
            .map(|&edges| {
                let diff = edges as f64 - avg_degree;
                diff * diff
            })
            .sum::<f64>() / node_clusters.len() as f64;

        (node_clusters.len(), var_degree)
    }
    // Degree Centrality 
    fn degree_centrality(&self, node: usize) -> f64 {
        if let Some(neighbors) = self.edges.get(&node) {
            let degree = neighbors.len() as f64;
            let total_nodes = (self.edges.len() - 1) as f64;  // Subtract 1 to exclude the node itself
            degree / total_nodes
        } else {
            0.0
        }
    }
    // Inside the `average_degree_centrality` function
    fn average_degree_centrality(&self) -> f64 {
        let total_centrality: f64 = self.edges.keys().map(|&node| self.degree_centrality(node)).sum();
        total_centrality / self.edges.keys().len() as f64
    }
    // Maximum Degree Centrality
    fn nodes_with_maximum_degree_centrality(&self) -> Vec<(usize, f64)> {
        let max_centrality = self.edges.keys()
            .map(|&node| self.degree_centrality(node))
            .fold(f64::NEG_INFINITY, f64::max);

        self.edges.keys()
            .filter(|&node| (self.degree_centrality(*node) - max_centrality).abs() < f64::EPSILON)
            .map(|&node| (node, self.degree_centrality(node)))
            .collect()
    }
    // Minimum Degree Centrality
    fn nodes_with_minimum_degree_centrality(&self) -> Vec<(usize, f64)> {
        let min_centrality = self.edges.keys()
            .map(|&node| self.degree_centrality(node))
            .fold(f64::INFINITY, f64::min);

        self.edges.keys()
            .filter(|&node| (self.degree_centrality(*node) - min_centrality).abs() < f64::EPSILON)
            .map(|&node| (node, self.degree_centrality(node)))
            .collect()
    }
    // End of measures 
    // Load graph data from a file
    fn load_graph_from_file(file_path: &str) -> io::Result<Graph> {
        let mut graph = Graph::new();

        let file = File::open(file_path)?;
        let reader = io::BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if !line.starts_with('#') {
                let mut iter = line.split_whitespace();
                if let (Some(from_str), Some(to_str)) = (iter.next(), iter.next()) {
                    if let (Ok(from), Ok(to)) = (from_str.parse::<usize>(), to_str.parse::<usize>()) {
                        graph.add_edge(from, to);
                    }
                }
            }
        }
        Ok(graph)
    }
}

fn main() {
    let file_path = "Amazon0302.txt";
    // Graph
    match Graph::load_graph_from_file(file_path) {
        Ok(graph) => {
            // Analysis on the graph
            let degree_distribution = graph.degree_distribution();
            println!("Degree Distribution: {:?}", degree_distribution);

            // Clustering
            let clustering_result = graph.graph_clustering();
            println!("Graph Clustering: {:?}", clustering_result);

            // Density and Average Degree
            let density = graph.graph_density();
            let average_degree = graph.average_degree();

            // Print Density and Average Degree
            println!("Density of the Graph: {}", density);
            println!("Average Degree: {}", average_degree);

            // Degree centrality measures
            let avg_degree_centrality = graph.average_degree_centrality();
            let max_degree_centrality_nodes = graph.nodes_with_maximum_degree_centrality();
            let min_degree_centrality_nodes = graph.nodes_with_minimum_degree_centrality();

            // Degree centrality measures Print
            println!("Average Degree Centrality: {:?}", avg_degree_centrality);
            // Nodes with Maximum Degree Centrality
            if !max_degree_centrality_nodes.is_empty() {
                println!("Nodes with Maximum Degree Centrality:");
                for &(node, centrality) in &max_degree_centrality_nodes {
                    println!("Node {}: Centrality {:?}", node, centrality);
                }
            }
            // Nodes with Minimum Degree Centrality
            if !min_degree_centrality_nodes.is_empty() {
                println!("Nodes with Minimum Degree Centrality:");
                for &(node, centrality) in &min_degree_centrality_nodes {
                    println!("Node {}: Centrality {:?}", node, centrality);
                }
            }
        }
        Err(e) => eprintln!("Error loading graph: {}", e),
    }
}

// Testing of Functions 
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new_graph_is_empty() {
        let graph = Graph::new();
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_add_edge() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);

        assert_eq!(graph.edges.len(), 3);
        assert_eq!(graph.edges[&1], vec![2, 3]);
        assert_eq!(graph.edges[&2], vec![]);
        assert_eq!(graph.edges[&3], vec![]);
    }

    #[test]
    fn test_graph_density() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        let density = graph.graph_density();
        assert_eq!(density, 0.5);
    }

    #[test]
    fn test_average_degree() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        let average_degree = graph.average_degree();
        assert_eq!(average_degree, 1.0); 
    }

    #[test]
    fn test_degree_distribution() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        let degree_distribution = graph.degree_distribution();
        let mut expected_distribution = HashMap::new();
        expected_distribution.insert(0, 1);
        expected_distribution.insert(1, 2);
        expected_distribution.insert(2, 1);
        println!("Actual Degree Distribution: {:?}", degree_distribution);
        assert_eq!(degree_distribution, expected_distribution);
    }

    #[test]
    fn test_graph_clustering() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        let (num_clusters, var_degree) = graph.graph_clustering();
        assert_eq!(num_clusters, 3);
        assert_eq!(var_degree, 0.0); 
    }

    #[test]
    fn test_average_degree_centrality() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        let avg_centrality = graph.average_degree_centrality();
        println!("Actual average degree centrality: {}", avg_centrality);
        assert_eq!(avg_centrality, 0.5); 
    }

    #[test]
    fn test_nodes_with_maximum_degree_centrality() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);

        let max_centrality_nodes = graph.nodes_with_maximum_degree_centrality();
        assert_eq!(max_centrality_nodes.len(), 1);
        assert_eq!(max_centrality_nodes[0].0, 1);
        assert_eq!(max_centrality_nodes[0].1, 1.0);
    }

    #[test]
    fn test_nodes_with_minimum_degree_centrality() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        let min_centrality_nodes = graph.nodes_with_minimum_degree_centrality();

        assert_eq!(min_centrality_nodes.len(), 1);
        assert_eq!(min_centrality_nodes[0].0, 4);
        assert_eq!(min_centrality_nodes[0].1, 0.0); // Update with the correct expected value
    }

    #[test]
    fn test_load_graph_from_file() {
        let file_path = "test_graph.txt";
        let test_content = "1 2\n2 3\n3 1\n";
        std::fs::write(file_path, test_content).expect("Failed to write test file");
        match Graph::load_graph_from_file(file_path) {
            Ok(graph) => {
                assert_eq!(graph.edges.len(), 3);
                assert_eq!(graph.edges[&1], vec![2]);
                assert_eq!(graph.edges[&2], vec![3]);
                assert_eq!(graph.edges[&3], vec![1]);
            }
            Err(e) => panic!("Error loading graph: {}", e),
        }
        std::fs::remove_file(file_path).expect("Failed to remove test file");
    }
}

