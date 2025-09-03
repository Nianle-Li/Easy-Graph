import math
import easygraph as eg
from easygraph.utils import *
from easygraph.utils.decorators import *

__all__ = ["eigenvector_centrality", "eigenvector_centrality_numpy"]

@not_implemented_for("multigraph")
@hybrid("cpp_eigenvector_centrality")
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
    """Compute the eigenvector centrality for the graph G.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node i is the
    i-th element of the vector x defined by the equation
    
    A x = λ x
    
    where A is the adjacency matrix of the graph G and λ is the largest
    eigenvalue. The centrality is normalized so that the sum of the squares
    equals 1.

    Parameters
    ----------
    G : graph
        A EasyGraph graph instance.

    max_iter : integer, optional (default=100)
        Maximum number of iterations in power method.

    tol : float, optional (default=1.0e-6)
        Tolerance for convergence test in power method.

    nstart : dictionary, optional (default=None)
        Starting value of eigenvector iteration for each node.
        If None, a starting vector is chosen randomly.

    weight : None or string, optional (default=None)
        If None, all edge weights are considered equal.
        Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    nodes : list
        List of eigenvector centrality values for each node.

    Raises
    ------
    EasyGraphPointlessConcept
        If the graph G is empty.

    EasyGraphError
        If the initial vector cannot have all zero values.

    EasyGraphNotImplemented
        If the algorithm fails to converge in max_iter iterations.

    Notes
    -----
    The measure was introduced by [1]_ and is discussed in [2]_.

    The eigenvector calculation is done by the power iteration method and has
    no guarantee of convergence. The iteration will stop after an error
    tolerance of len(G) * tol has been reached. If the number of iterations
    exceed max_iter then a EasyGraphNotImplemented exception is raised.

    For directed graphs this returns the "right" eigenvector centrality
    corresponding to the out-edges in the graph. For "left" eigenvector
    centrality corresponding to the in-edges see
    numpy.linalg.eig function.

    For MultiGraph/MultiDiGraph, the edges weights are aggregated
    before the calculation.

    Examples
    --------
    >>> G = eg.path_graph(4)
    >>> centrality = eg.eigenvector_centrality(G)
    >>> len(centrality)
    4

    References
    ----------
    .. [1] Phillip Bonacich.
       Power and Centrality: A Family of Measures.
       American Journal of Sociology 92(5):1170–1182, 1986
       http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf
    .. [2] Mark E. J. Newman.
       Networks: An Introduction.
       Oxford University Press, USA, 2010, pp. 169.
    """
    if len(G) == 0:
        raise eg.EasyGraphPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    
    # If no initial vector is provided, start with the all-ones vector.
    if nstart is None:
        nstart = {n: 1 for n in G}
    if all(v == 0 for v in nstart.values()):
        raise eg.EasyGraphError("initial vector cannot have all zero values")
    
    # Normalize the initial vector so that each entry is in [0, 1]. This is
    # guaranteed to never have a divide-by-zero error by the previous line.
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    nnodes = G.number_of_nodes()
    
    # make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = xlast.copy()  # Start with xlast times I to iterate with (A+I)
        # do the multiplication y^T = x^T A (left eigenvector)
        for n in x:
            for nbr in G[n]:
                w = G[n][nbr].get(weight, 1) if weight else 1
                x[nbr] += xlast[n] * w
        
        # Normalize the vector. The normalization denominator `norm`
        # should never be zero by the Perron--Frobenius
        # theorem. However, in case it is due to numerical error, we
        # assume the norm to be one instead.
        norm = math.hypot(*x.values()) or 1
        x = {k: v / norm for k, v in x.items()}
        
        # Check for convergence (in the L_1 norm).
        if sum(abs(x[n] - xlast[n]) for n in x) < nnodes * tol:
            # Convert to list format consistent with other centrality functions
            result = [0.0 for _ in range(len(G))]
            for i in range(len(result)):
                result[i] = x[G.index2node[i]]
            return result
    
    raise eg.EasyGraphNotImplemented(
        f"eigenvector centrality: power iteration failed to converge "
        f"within {max_iter} iterations"
    )
@not_implemented_for("multigraph")
@hybrid("cpp_eigenvector_centrality_numpy")
def eigenvector_centrality_numpy(G, weight=None, max_iter=50, tol=0):
    """Compute the eigenvector centrality for the graph G using NumPy/SciPy.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node i is the
    i-th element of a left eigenvector associated with the eigenvalue λ
    of maximum modulus that is positive.

    This implementation uses SciPy sparse eigenvalue solver (ARPACK)
    to find the largest eigenvalue/eigenvector pair.

    Parameters
    ----------
    G : graph
        A connected EasyGraph graph.

    weight : None or string, optional (default=None)
        If None, all edge weights are considered equal. Otherwise holds the
        name of the edge attribute used as weight. In this measure the
        weight is interpreted as the connection strength.

    max_iter : integer, optional (default=50)
        Maximum number of Arnoldi update iterations allowed.

    tol : float, optional (default=0)
        Relative accuracy for eigenvalues (stopping criterion).
        The default value of 0 implies machine precision.

    Returns
    -------
    nodes : list
        List of eigenvector centrality values for each node.
        The associated vector has unit Euclidean norm and the values are
        nonnegative.

    Raises
    ------
    EasyGraphPointlessConcept
        If the graph G is the null graph.

    EasyGraphError
        When the requested convergence is not obtained or if G is not connected.

    Examples
    --------
    >>> G = eg.path_graph(4)
    >>> centrality = eg.eigenvector_centrality_numpy(G)
    >>> len(centrality)
    4

    Notes
    -----
    This implementation uses the SciPy sparse eigenvalue solver (ARPACK)
    to find the largest eigenvalue/eigenvector pair using Arnoldi iterations.

    For disconnected graphs, this function may give inconsistent results
    due to the non-uniqueness of the eigenvector. Only connected graphs
    are recommended for consistent results.
    """
    try:
        import numpy as np
        import scipy as sp
        from scipy.sparse import linalg as spla
    except ImportError:
        raise ImportError(
            "eigenvector_centrality_numpy requires NumPy and SciPy"
        )

    if len(G) == 0:
        raise eg.EasyGraphPointlessConcept(
            "cannot compute centrality for the null graph"
        )

    # Check connectivity for consistent results
    if G.is_directed():
        connected = eg.is_strongly_connected(G)
        if not connected:
            raise eg.EasyGraphError(
                "eigenvector_centrality_numpy does not give consistent results "
                "for disconnected directed graphs"
            )
    else:
        connected = eg.is_connected(G)
        if not connected:
            raise eg.EasyGraphError(
                "eigenvector_centrality_numpy does not give consistent results "
                "for disconnected graphs"
            )

    # Create node list in consistent order
    nodelist = list(G.nodes())
    
    # Convert to scipy sparse matrix
    try:
        # Try to use EasyGraph's built-in conversion if available
        if hasattr(eg, 'to_scipy_sparse_array'):
            M = eg.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
        else:
            # Fallback: manual construction
            M = _graph_to_scipy_sparse(G, nodelist=nodelist, weight=weight)
    except Exception:
        # Manual matrix construction as fallback
        M = _graph_to_scipy_sparse(G, nodelist=nodelist, weight=weight)

    try:
        # Compute largest eigenvalue and eigenvector
        # Use transpose for left eigenvector (standard for centrality)
        eigenvalues, eigenvectors = spla.eigs(
            M.T, k=1, which="LR", maxiter=max_iter, tol=tol
        )
        
        # Extract the largest eigenvector
        largest = eigenvectors.flatten().real
        
        # Normalize the eigenvector
        # Ensure all values are non-negative by flipping sign if needed
        if largest.sum() < 0:
            largest = -largest
        
        # Normalize to unit norm
        norm = sp.linalg.norm(largest)
        if norm > 0:
            largest = largest / norm
        
        # Convert to list format consistent with EasyGraph style
        result = [0.0 for _ in range(len(G))]
        for i, node in enumerate(nodelist):
            node_idx = G.node2index[node]
            result[node_idx] = float(largest[i])
        
        return result
        
    except spla.ArpackNoConvergence as e:
        raise eg.EasyGraphError(
            f"eigenvector_centrality_numpy failed to converge: {e}"
        )
    except Exception as e:
        raise eg.EasyGraphError(
            f"eigenvector_centrality_numpy computation failed: {e}"
        )


def _graph_to_scipy_sparse(G, nodelist=None, weight=None):
    """Convert EasyGraph graph to SciPy sparse matrix.
    
    This is a helper function for eigenvector_centrality_numpy when
    the built-in conversion is not available.
    """
    import numpy as np
    from scipy import sparse
    
    if nodelist is None:
        nodelist = list(G.nodes())
    
    n = len(nodelist)
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    
    # Collect edges and weights
    row_indices = []
    col_indices = []
    data = []
    
    for u in G:
        u_idx = node_to_idx[u]
        for v in G[u]:
            v_idx = node_to_idx[v]
            w = G[u][v].get(weight, 1) if weight else 1
            row_indices.append(u_idx)
            col_indices.append(v_idx)
            data.append(float(w))
    
    # Create sparse matrix
    M = sparse.csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(n, n), 
        dtype=float
    )
    
    return M



def compare_eigenvector_centrality_speed(n=100, m=200):
    """
    比较 EasyGraph (Python 版本和 C++混编) 与 NetworkX 计算特征向量中心性的时间。
    
    Parameters
    ----------
    n : int, optional (default=100)
        节点数
    m : int, optional (default=200)
        边数
    
    Returns
    -------
    dict
        包含各种方法的计算时间: 
        {'easygraph_cpp_time': ..., 'easygraph_python_time': ..., 'networkx_time': ...}
    """
    import time
    import easygraph as eg
    import networkx as nx
    import os

    # 用 networkx 生成随机图，然后转换为 EasyGraph
    G_nx = nx.gnm_random_graph(n, m)
    G_eg = eg.from_networkx(G_nx)
    
    # 设置环境变量来控制是否使用 C++ 版本
    results = {}
    
    # 1. EasyGraph C++ 混编版本计时
    os.environ["EASYGRAPH_USE_CPP"] = "1"
    G_eg.cflag = True 
    t0 = time.time()
    eg.eigenvector_centrality(G_eg)
    t1 = time.time()
    results["easygraph_cpp_time"] = t1 - t0
    
    # 2. EasyGraph Python 版本计时
    os.environ["EASYGRAPH_USE_CPP"] = "0"
    t0 = time.time()
    eg.eigenvector_centrality(G_eg)
    t1 = time.time()
    results["easygraph_python_time"] = t1 - t0
    
    # 3. NetworkX 计时
    t0 = time.time()
    nx.eigenvector_centrality(G_nx)
    t1 = time.time()
    results["networkx_time"] = t1 - t0
    
    # 恢复环境变量默认值（如果有必要）
    os.environ.pop("EASYGRAPH_USE_CPP", None)
    
    # 打印结果进行比较
    print(f"节点数: {n}, 边数: {m}")
    print(f"EasyGraph (C++): {results['easygraph_cpp_time']:.6f} 秒")
    print(f"EasyGraph (Python): {results['easygraph_python_time']:.6f} 秒")
    print(f"NetworkX: {results['networkx_time']:.6f} 秒")
    
    # 计算加速比
    if results['networkx_time'] > 0:
        print(f"EasyGraph (C++) 相比 NetworkX 加速比: {results['networkx_time'] / results['easygraph_cpp_time']:.2f}x")
    if results['easygraph_python_time'] > 0:
        print(f"EasyGraph (C++) 相比 EasyGraph (Python) 加速比: {results['easygraph_python_time'] / results['easygraph_cpp_time']:.2f}x")
    
    return results

def test_cpp_function():
    """直接测试C++函数是否可用"""
    import easygraph as eg
    import networkx as nx
    
    # 首先尝试导入cpp_easygraph模块
    try:
        import cpp_easygraph
        print("成功导入cpp_easygraph模块")
    except ImportError as e:
        print(f"错误: 无法导入cpp_easygraph模块 - {e}")
        print("这可能意味着C++扩展未正确编译或安装")
        return
    
    # 创建一个简单图
    G_nx = nx.path_graph(4)
    G_eg = eg.from_networkx(G_nx)
    G_eg.cflag = True  # 重要：设置cflag
    
    # 检查cpp_easygraph模块
    print("cpp_easygraph中可用的函数:")
    cpp_funcs = [item for item in dir(cpp_easygraph) if item.startswith("cpp_")]
    if cpp_funcs:
        for item in cpp_funcs:
            print(f" - {item}")
    else:
        print("警告: 在cpp_easygraph中找不到任何以'cpp_'开头的函数")
    
    # 特别检查我们关心的函数
    if hasattr(cpp_easygraph, "cpp_eigenvector_centrality"):
        print("找到cpp_eigenvector_centrality函数")
        
        # 尝试直接调用
        try:
            print("尝试直接调用cpp_eigenvector_centrality...")
            result = cpp_easygraph.cpp_eigenvector_centrality(G_eg, None, None, None, None)
            print(f"调用成功，结果: {result}")
        except Exception as e:
            print(f"调用失败: {type(e).__name__}: {e}")
    else:
        print("错误: cpp_eigenvector_centrality函数不存在!")