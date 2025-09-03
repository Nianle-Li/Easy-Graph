from easygraph.functions.centrality.eigenvector import compare_eigenvector_centrality_speed, test_cpp_function
import easygraph as eg
import networkx as nx

# 比较 EasyGraph 与 NetworkX 的计算速度
speed_result = compare_eigenvector_centrality_speed(n=100, m=200)
# 注意: 函数已经包含了结果打印，以下是额外展示
print("\n结果总结:")
print(f"EasyGraph (C++): {speed_result['easygraph_cpp_time']:.6f} 秒")
print(f"EasyGraph (Python): {speed_result['easygraph_python_time']:.6f} 秒") 
print(f"NetworkX: {speed_result['networkx_time']:.6f} 秒")

# 计算和显示加速比
if speed_result['networkx_time'] > 0 and speed_result['easygraph_cpp_time'] > 0:
    nx_cpp_speedup = speed_result['networkx_time'] / speed_result['easygraph_cpp_time']
    print(f"C++版本比NetworkX快 {nx_cpp_speedup:.2f} 倍")

if speed_result['easygraph_python_time'] > 0 and speed_result['easygraph_cpp_time'] > 0:
    py_cpp_speedup = speed_result['easygraph_python_time'] / speed_result['easygraph_cpp_time']
    print(f"C++版本比Python版本快 {py_cpp_speedup:.2f} 倍")

# 测试 C++ 函数
test_cpp_function()

