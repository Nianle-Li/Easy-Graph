#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "centrality.h"
#include "../../classes/graph.h"

namespace py = pybind11;

py::object cpp_eigenvector_centrality(
    py::object G,
    py::object py_max_iter,
    py::object py_tol,
    py::object py_nstart,
    py::object py_weight
) {
    try {
        Graph& graph = G.cast<Graph&>();
        int max_iter = py_max_iter.cast<int>();
        double tol = py_tol.cast<double>();
        
        // 获取权重键(如果提供)
        std::string weight_key = "";
        if (!py_weight.is_none()) {
            weight_key = py_weight.cast<std::string>();
        }
        
        // 生成CSR表示
        auto csr = graph.gen_CSR(weight_key);
        int n = csr->nodes.size();
        
        if (n == 0) {
            return py::dict();  //问题：可能和python版本不一致
        }
        
        // 初始化向量 
        std::vector<double> x0(n, 0.0);
        std::vector<double> x1(n, 0.0);
        std::vector<double>* x_prev = &x0;
        std::vector<double>* x_next = &x1;
        
        // 处理nstart参数
        if (py_nstart.is_none()) {
            // 默认情况：所有值设为1
            for (int i = 0; i < n; i++) {
                (*x_prev)[i] = 1.0;
            }
        } else {
            // 使用提供的起始向量
            py::dict nstart = py_nstart.cast<py::dict>();
            double sum = 0.0;
            
            for (int i = 0; i < n; i++) {
                node_t node_id = csr->nodes[i];
                py::object node_obj = graph.id_to_node[py::cast(node_id)];
                if (nstart.contains(node_obj)) {
                    (*x_prev)[i] = nstart[node_obj].cast<double>();
                }
                sum += (*x_prev)[i];
            }
            
            // 检查所有值是否为零
            if (sum == 0.0) {
                throw std::runtime_error("initial vector cannot have all zero values");
            }
            
            // 归一化
            for (int i = 0; i < n; i++) {
                (*x_prev)[i] /= sum;
            }
        }
        
        // 幂迭代法
        for (int iter = 0; iter < max_iter; iter++) {
            // 保存上一次迭代结果
            std::swap(x_prev, x_next);
            std::fill((*x_next).begin(), (*x_next).end(), 0.0);
            
            // 计算 x = A * xlast
            for (int i = 0; i < n; i++) {
                int start = csr->V[i];
                int end = csr->V[i + 1];
                
                for (int jj = start; jj < end; jj++) {
                    int j = csr->E[jj];
                    double w = 1.0;
                    
                    // 应用权重（如果指定）
                    if (!weight_key.empty() && 
                        csr->W_map.find(weight_key) != csr->W_map.end()) {
                        w = (*csr->W_map[weight_key])[jj];
                    }
                    
                    // 在邻接表中找到目标节点索引
                    for (int k = 0; k < n; k++) {
                        if (csr->nodes[k] == j) {
                            (*x_next)[k] += (*x_prev)[i] * w;
                            break;
                        }
                    }
                }
            }
            
            // 归一化向量 
            std::vector<double>& x_final = *x_next;
            double norm = 0.0;
            for (double val : x_final) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            
            if (norm > 0) {
                for (int i = 0; i < n; i++) {
                    x_final[i] /= norm;
                }
            }
            
            // 检查收敛性
            double diff = 0.0;
            for (int i = 0; i < n; i++) {
                diff += std::abs((*x_next)[i] - (*x_prev)[i]);
            }
            
            if (diff < n * tol) {
                // 已收敛 - 准备结果
                py::dict result;
                for (int i = 0; i < n; i++) {
                    node_t internal_id = csr->nodes[i];
                    py::object node_obj = graph.id_to_node[py::cast(internal_id)];
                    result[node_obj] = (*x_next)[i];
                }
                return result;
            }
        }
        
        // 如果到这里，说明没有收敛
        throw std::runtime_error("eigenvector centrality: power iteration failed to converge "
                               "within " + std::to_string(max_iter) + " iterations");
        
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

py::object cpp_eigenvector_centrality_numpy(
    py::object G,
    py::object py_weight,
    py::object py_max_iter,
    py::object py_tol
) {
    try {
        // 导入numpy和scipy以调用Python函数
        py::module numpy = py::module::import("numpy");
        py::module scipy_sparse = py::module::import("scipy.sparse");
        py::module scipy_sparse_linalg = py::module::import("scipy.sparse.linalg");
        
        Graph& graph = G.cast<Graph&>();
        int max_iter = py_max_iter.cast<int>();
        double tol = py_tol.cast<double>();
        
        // 获取权重键(如果提供)
        std::string weight_key = "";
        if (!py_weight.is_none()) {
            weight_key = py_weight.cast<std::string>();
        }
        
        // 生成CSR表示
        auto csr = graph.gen_CSR(weight_key);
        int n = csr->nodes.size();
        
        if (n == 0) {
            throw std::runtime_error("cannot compute centrality for the null graph");
        }
        
        // 检查连通性
        bool connected = false;
        if (G.attr("is_directed")().cast<bool>()) {
            py::object result = py::module::import("easygraph").attr("is_strongly_connected")(G);
            connected = result.cast<bool>();
            if (!connected) {
                throw std::runtime_error("eigenvector_centrality_numpy does not give consistent results for disconnected directed graphs");
            }
        } else {
            py::object result = py::module::import("easygraph").attr("is_connected")(G);
            connected = result.cast<bool>();
            if (!connected) {
                throw std::runtime_error("eigenvector_centrality_numpy does not give consistent results for disconnected graphs");
            }
        }
        
        // 创建节点列表，保持一致的顺序
        std::vector<py::object> nodelist;
        for (int i = 0; i < n; i++) {
            node_t internal_id = csr->nodes[i];
            py::object node_obj = graph.id_to_node[py::cast(internal_id)];
            nodelist.push_back(node_obj);
        }
        
        // 创建稀疏矩阵 (CSR格式)
        std::vector<int> row_indices;
        std::vector<int> col_indices;
        std::vector<double> data;
        
        // 填充稀疏矩阵数据
        for (int i = 0; i < n; i++) {
            int start = csr->V[i];
            int end = csr->V[i + 1];
            
            for (int jj = start; jj < end; jj++) {
                int j = csr->E[jj];
                double w = 1.0;
                
                // 应用权重（如果指定）
                if (!weight_key.empty() && 
                    csr->W_map.find(weight_key) != csr->W_map.end()) {
                    w = (*csr->W_map[weight_key])[jj];
                }
                
                // 找到目标节点在nodelist中的索引
                for (int k = 0; k < n; k++) {
                    node_t node_id = csr->nodes[k];
                    if (node_id == j) {
                        row_indices.push_back(i);
                        col_indices.push_back(k);
                        data.push_back(w);
                        break;
                    }
                }
            }
        }
        
        // 使用scipy创建稀疏矩阵
        py::array_t<int> py_row_indices = py::array_t<int>(row_indices.size(), row_indices.data());
        py::array_t<int> py_col_indices = py::array_t<int>(col_indices.size(), col_indices.data());
        py::array_t<double> py_data = py::array_t<double>(data.size(), data.data());
        
        py::object sparse_matrix = scipy_sparse.attr("csr_matrix")(
            py::make_tuple(py_data, py::make_tuple(py_row_indices, py_col_indices)),
            py::make_tuple(n, n)
        );
        
        // 转置矩阵用于计算左特征向量
        py::object M = sparse_matrix.attr("transpose")();
        
        try {
            // 使用ARPACK求解最大特征值和特征向量
            py::object eigenvalues_eigenvectors = scipy_sparse_linalg.attr("eigs")(
                M, py::int_(1), py::str("LR"), py::none(),
                py::none(), py::bool_(false), py::none(),
                max_iter, tol, py::none()
            );
            
            py::object eigenvectors = eigenvalues_eigenvectors.attr("__getitem__")(1);
            py::object largest = eigenvectors.attr("flatten")().attr("real");
            
            // 提取特征向量并归一化
            py::array_t<double> largest_array = largest.cast<py::array_t<double>>();
            py::buffer_info buf_info = largest_array.request();
            double* ptr = static_cast<double*>(buf_info.ptr);
            
            // 确保所有值非负 (如果需要，翻转符号)
            double sum_val = 0.0;
            for (size_t i = 0; i < buf_info.size; i++) {
                sum_val += ptr[i];
            }
            
            if (sum_val < 0) {
                for (size_t i = 0; i < buf_info.size; i++) {
                    ptr[i] = -ptr[i];
                }
            }
            
            // 归一化为单位范数
            double norm = numpy.attr("linalg").attr("norm")(largest_array).cast<double>();
            if (norm > 0) {
                for (size_t i = 0; i < buf_info.size; i++) {
                    ptr[i] /= norm;
                }
            }
            
            // 转换为EasyGraph风格的结果列表
            std::vector<double> result(n, 0.0);
            for (int i = 0; i < n; i++) {
                node_t node_id = csr->nodes[i];
                int node_idx = graph.node_to_id.attr("__getitem__")(nodelist[i]).cast<int>();
                result[node_idx] = ptr[i];
            }
            
            // 创建numpy数组返回
            py::array_t<double> ret = py::array_t<double>(result.size());
            py::buffer_info ret_buf = ret.request();
            double* ret_ptr = static_cast<double*>(ret_buf.ptr);
            
            for (size_t i = 0; i < result.size(); i++) {
                ret_ptr[i] = result[i];
            }
            
            return ret;
            
        } catch (const py::error_already_set& e) {
            // 捕获ArpackNoConvergence错误
            if (std::string(e.what()).find("ArpackNoConvergence") != std::string::npos) {
                throw std::runtime_error("eigenvector_centrality_numpy failed to converge");
            }
            throw std::runtime_error("eigenvector_centrality_numpy computation failed: " + std::string(e.what()));
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}
