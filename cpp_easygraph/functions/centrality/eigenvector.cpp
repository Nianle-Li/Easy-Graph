#include "centrality.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

// 添加必要的 pybind11 头文件
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../../classes/graph.h"
#include "../../common/utils.h"

namespace py = pybind11;

// 辅助函数：判断是否为多重图
inline bool is_multigraph(Graph* G) {
    // Graph_is_multigraph 返回 py::object，可转 bool
    return py::cast<bool>(Graph_is_multigraph(py::cast(G)));
}

// 辅助函数：判断是否为有向图
inline bool is_directed(Graph* G) {
    // Graph_is_directed 返回 py::object，可转 bool
    return py::cast<bool>(Graph_is_directed(py::cast(G)));
}

std::vector<double> eigenvector_centrality_impl(
    Graph* G,
    int max_iter,
    double tol,
    std::unordered_map<int, double> nstart,
    const std::string& weight) {
    
    // Check for empty graph
    if (G->node.size() == 0) {
        throw std::runtime_error("cannot compute centrality for the null graph");
    }
    
    // Check for multigraph
    if (is_multigraph(G)) {
        throw std::runtime_error("eigenvector centrality not implemented for multigraphs");
    }

    // If no initial vector is provided, start with the all-ones vector
    if (nstart.empty()) {
        for (int i = 0; i < G->node.size(); i++) {
            nstart[i] = 1.0;
        }
    }
    
    // Check if initial vector has all zero values
    bool all_zeros = true;
    for (const auto& pair : nstart) {
        if (pair.second != 0) {
            all_zeros = false;
            break;
        }
    }
    if (all_zeros) {
        throw std::runtime_error("initial vector cannot have all zero values");
    }
    
    // Normalize the initial vector
    double nstart_sum = 0.0;
    for (const auto& pair : nstart) {
        nstart_sum += pair.second;
    }
    
    std::unordered_map<int, double> x;
    for (const auto& pair : nstart) {
        x[pair.first] = pair.second / nstart_sum;
    }
    
    int nnodes = G->node.size();
    
    // Make up to max_iter iterations
    for (int i = 0; i < max_iter; i++) {
        std::unordered_map<int, double> xlast = x;
        // 修正：x应该被清零，而不是保留xlast的值
        // 这与Python版本 x = xlast.copy() 后面的逻辑一致
        for (auto& pair : x) {
            pair.second = 0.0;
        }
        
        // do the multiplication y^T = x^T A (left eigenvector)
        for (int n = 0; n < G->node.size(); n++) {
            for (const auto& nbr_data : G->adj[n]) {
                int nbr = nbr_data.first;
                double w = 1.0;
                
                // 修正：直接访问 adj[n][nbr] 字典
                if (!weight.empty()) {
                    const auto& edge_attr = nbr_data.second;
                    auto it = edge_attr.find(weight);
                    if (it != edge_attr.end()) {
                        w = it->second;
                    }
                }
                
                x[nbr] += xlast[n] * w;
            }
        }
        
        // Calculate normalization factor (Euclidean norm)
        double norm = 0.0;
        for (const auto& pair : x) {
            norm += pair.second * pair.second;
        }
        norm = std::sqrt(norm);
        
        // Use norm = 1.0 if norm is zero due to numerical error
        if (norm == 0.0) {
            norm = 1.0;
        }
        
        // Normalize the vector
        for (auto& pair : x) {
            pair.second /= norm;
        }
        
        // Check for convergence (in the L_1 norm)
        double error = 0.0;
        for (const auto& pair : x) {
            error += std::abs(pair.second - xlast[pair.first]);
        }
        
        if (error < nnodes * tol) {
            // Convert to list format consistent with other centrality functions
            std::vector<double> result(nnodes, 0.0);
            for (int i = 0; i < nnodes; i++) {
                result[i] = x[i];
            }
            return result;
        }
    }
    
    throw std::runtime_error(
        "eigenvector centrality: power iteration failed to converge within " + 
        std::to_string(max_iter) + " iterations"
    );
}

std::vector<double> eigenvector_centrality_numpy_impl(
    Graph* G,
    const std::string& weight,
    int max_iter,
    double tol) {
    
    // Check for empty graph
    if (G->node.size() == 0) {
        throw std::runtime_error("cannot compute centrality for the null graph");
    }
    
    // Check for multigraph
    if (is_multigraph(G)) {
        throw std::runtime_error("eigenvector centrality not implemented for multigraphs");
    }

    // Check connectivity for consistent results
    bool connected = false;
    if (is_directed(G)) {
        // This would require implementing is_strongly_connected in C++
        // For now, we'll throw an error
        throw std::runtime_error(
            "eigenvector_centrality_numpy does not support directed graphs in C++ implementation yet"
        );
    } else {
        // This would require implementing is_connected in C++
        // For now, we'll assume it's connected
        connected = true;
    }
    
    if (!connected) {
        throw std::runtime_error(
            "eigenvector_centrality_numpy does not give consistent results "
            "for disconnected graphs"
        );
    }
    
    // Note: In a real implementation, we would use a C++ linear algebra library
    // such as Eigen or Armadillo to compute the largest eigenvalue/eigenvector pair.
    // For now, we'll fall back to the power iteration method
    
    // Create initial vector with all ones
    std::unordered_map<int, double> nstart;
    for (int i = 0; i < G->node.size(); i++) {
        nstart[i] = 1.0;
    }
    
    // Call the power iteration implementation
    return eigenvector_centrality_impl(G, max_iter, tol, nstart, weight);
}

// 重写接口函数，完全参照 closeness.cpp 的模式
static py::object invoke_cpp_eigenvector_centrality(
    py::object G,
    py::object py_max_iter,
    py::object py_tol,
    py::object py_nstart,
    py::object py_weight) {
    
    // 直接转换 Graph 对象 - 与 closeness.cpp 完全一致
    Graph& G_ = G.cast<Graph&>();
    
    // 处理参数 - 与 closeness.cpp 完全一致的模式
    int max_iter = py_max_iter.is_none() ? 100 : py_max_iter.cast<int>();
    double tol = py_tol.is_none() ? 1.0e-6 : py_tol.cast<double>();
    std::string weight_key = weight_to_string(py_weight);
    
    // 处理起始向量 - 遵循 closeness.cpp 的错误处理模式
    std::unordered_map<int, double> nstart;
    if (!py_nstart.is_none()) {
        py::dict nstart_dict = py_nstart;
        for (auto item : nstart_dict) {
            py::handle key_handle = item.first;
            py::object key = py::reinterpret_borrow<py::object>(key_handle);
            // 使用与 closeness.cpp 相同的节点查找和错误处理
            if (G_.node_to_id.attr("get")(key, py::none()).is_none()) {
                printf("Node not found in graph!");
                return py::none();
            }
            
            node_t node_id = G_.node_to_id.attr("get")(key).cast<node_t>();
            double value = py::cast<double>(item.second);
            nstart[node_id] = value;
        }
    }
    
    // 调用实现函数
    std::vector<double> result;
    try {
        result = eigenvector_centrality_impl(&G_, max_iter, tol, nstart, weight_key);
    } catch (const std::exception& e) {
        printf("Error in eigenvector_centrality: %s\n", e.what());
        return py::none();
    }
    
    // 创建返回数组 - 完全匹配 closeness.cpp
    py::array::ShapeContainer ret_shape{(int)result.size()};
    py::array_t<double> ret(ret_shape, result.data());
    
    return ret;
}

static py::object invoke_cpp_eigenvector_centrality_numpy(
    py::object G,
    py::object py_weight,
    py::object py_max_iter,
    py::object py_tol) {
    
    // 与 closeness.cpp 完全一致
    Graph& G_ = G.cast<Graph&>();
    
    // 处理参数
    int max_iter = py_max_iter.is_none() ? 50 : py_max_iter.cast<int>();
    double tol = py_tol.is_none() ? 0.0 : py_tol.cast<double>();
    std::string weight_key = weight_to_string(py_weight);
    
    // 调用实现函数 - 添加错误处理与 closeness.cpp 一致
    std::vector<double> result;
    try {
        result = eigenvector_centrality_numpy_impl(&G_, weight_key, max_iter, tol);
    } catch (const std::exception& e) {
        printf("Error in eigenvector_centrality_numpy: %s\n", e.what());
        return py::none();
    }
    
    // 创建返回数组 - 与 closeness.cpp 完全一致
    py::array::ShapeContainer ret_shape{(int)result.size()};
    py::array_t<double> ret(ret_shape, result.data());
    
    return ret;
}

// 暴露给Python的函数 - 完全匹配 closeness.cpp 和 betweenness.cpp
py::object cpp_eigenvector_centrality(
    py::object G,
    py::object py_max_iter,
    py::object py_tol,
    py::object py_nstart,
    py::object py_weight) {
    
    // 不用 try/catch，直接返回 - 与 closeness.cpp 完全一致
    return invoke_cpp_eigenvector_centrality(G, py_max_iter, py_tol, py_nstart, py_weight);
}

py::object cpp_eigenvector_centrality_numpy(
    py::object G,
    py::object py_weight,
    py::object py_max_iter,
    py::object py_tol) {
    
    // 不用 try/catch，直接返回 - 与 closeness.cpp 完全一致
    return invoke_cpp_eigenvector_centrality_numpy(G, py_weight, py_max_iter, py_tol);
}
