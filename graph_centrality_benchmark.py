import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import easygraph as eg
import inspect

def convert_to_graphs(nx_graph):
    """将NetworkX图转换为各个库的图对象"""
    # 提取边列表
    edges = list(nx_graph.edges())
    
    # iGraph
    ig_vertices = sorted(list(nx_graph.nodes()))
    ig_g = ig.Graph()
    ig_g.add_vertices(ig_vertices)
    ig_g.add_edges(edges)
    
    # Easy Graph
    eg_g = eg.Graph()  
    for u, v in edges:
        eg_g.add_edge(u, v)
    
    return {
        'networkx': nx_graph,
        'igraph': ig_g, 
        'easy_graph': eg_g,
    }

def check_eg_implementation():
    """检查EasyGraph是否使用了C++实现的特征向量中心性算法"""
    try:
        # 获取特征向量中心性函数的源代码
        source = inspect.getsource(eg.eigenvector_centrality)
        
        # 检查是否包含C++实现的调用
        cpp_call_present = "@hybrid(" in source
        
        if cpp_call_present:
            print("EasyGraph eigenvector_centrality 函数已配置为使用C++实现")
            print(f"调用路径: {source.split('@hybrid(')[1].split(')')[0]}")
            
            # 创建一个小图进行测试
            test_g = eg.Graph()
            test_g.add_edge(0, 1)
            test_g.add_edge(1, 2)
            test_g.add_edge(2, 0)
            
            # 设置环境变量以输出详细信息
            os.environ['EASYGRAPH_DEBUG'] = '1'
            
            try:
                # 尝试调用函数
                start = time.time()
                result = eg.eigenvector_centrality(test_g)
                end = time.time()
                
                print(f"调用完成，耗时: {end-start:.6f}秒")
                print(f"结果: {result}")
                
                # 检查是否实际调用了C++实现
                # 检查正确的C++函数路径
                eg_path = os.path.dirname(os.path.dirname(inspect.getfile(eg)))
                correct_cpp_path = "/users/sama/Easy-Graph/cpp_easygraph/functions/centrality"
                
                print(f"预期的EasyGraph C++路径: {correct_cpp_path}")
                
                if os.path.exists(correct_cpp_path):
                    print(f"EasyGraph C++路径存在: {correct_cpp_path}")
                    print("目录内容:")
                    for file in os.listdir(correct_cpp_path):
                        print(f"  - {file}")
                    
                    # 检查eigenvector.cpp文件
                    eigenvector_cpp = os.path.join(correct_cpp_path, "eigenvector.cpp")
                    if os.path.exists(eigenvector_cpp):
                        print(f"eigenvector.cpp文件存在")
                        # 分析文件内容检查函数实现
                        with open(eigenvector_cpp, 'r') as f:
                            content = f.read()
                            if "py::object cpp_eigenvector_centrality" in content:
                                print("C++文件中包含了正确的Python接口函数")
                            else:
                                print("警告：C++文件中可能缺少正确的Python接口函数")
                    else:
                        print(f"eigenvector.cpp文件不存在，这可能是问题所在")
                        
                    # 检查PYTHONPATH环境变量是否包含正确路径
                    pythonpath = os.environ.get('PYTHONPATH', '')
                    print(f"PYTHONPATH: {pythonpath}")
                    
                    # 测试C++模块是否可导入
                    try:
                        import sys
                        print("\n尝试直接导入C++模块:")
                        print(f"sys.path: {sys.path}")
                        
                        # 尝试导入不同可能的模块路径
                        modules_to_try = [
                            "cpp_easygraph.functions.centrality", 
                            "easygraph.cpp_easygraph.functions.centrality",
                            "easygraph.functions.centrality"
                        ]
                        
                        for module_path in modules_to_try:
                            try:
                                print(f"尝试导入 {module_path}")
                                module = __import__(module_path, fromlist=['*'])
                                if hasattr(module, 'cpp_eigenvector_centrality'):
                                    print(f"成功从 {module_path} 导入cpp_eigenvector_centrality")
                                else:
                                    print(f"模块 {module_path} 存在，但未找到cpp_eigenvector_centrality函数")
                            except ImportError as e:
                                print(f"无法导入 {module_path}: {e}")
                    except Exception as e:
                        print(f"测试模块导入时出错: {e}")
                else:
                    print(f"EasyGraph C++路径不存在: {correct_cpp_path}")
                    print("这可能是C++实现未被找到的原因")
                
                # 尝试获取Python代码中的hybrid装饰器配置
                try:
                    from easygraph.functions.centrality import eigenvector_centrality
                    eg_source = inspect.getsource(eigenvector_centrality)
                    print("\nPython eigenvector_centrality函数定义:")
                    hybrid_line = [line for line in eg_source.split('\n') if '@hybrid' in line]
                    if hybrid_line:
                        print(f"hybrid装饰器配置: {hybrid_line[0]}")
                        # 检查hybrid装饰器中的C++函数名是否正确
                        cpp_func_path = hybrid_line[0].split('"')[1]
                        print(f"C++函数路径: {cpp_func_path}")
                        
                        # 路径是否正确？
                        expected_path = "cpp_easygraph.functions.centrality.cpp_eigenvector_centrality"
                        if cpp_func_path == expected_path:
                            print("hybrid装饰器中的C++函数路径正确")
                        else:
                            print(f"警告：hybrid装饰器中的C++函数路径可能不正确，应为: {expected_path}")
                    else:
                        print("未找到hybrid装饰器配置")
                except ImportError:
                    print("无法直接导入eigenvector_centrality模块")
                except Exception as e:
                    print(f"检查Python代码时出错: {str(e)}")
                
                # 检查是否使用了Python或C++实现
                print("\n检查是否成功使用C++实现:")
                # 尝试临时禁用C++实现，测试运行时间差异
                try:
                    # 保存当前环境变量设置
                    old_env = os.environ.get('EASYGRAPH_CPP_ACCELERATOR', None)
                    
                    # 强制使用Python实现
                    os.environ['EASYGRAPH_CPP_ACCELERATOR'] = 'OFF'
                    
                    start_py = time.time()
                    py_result = eg.eigenvector_centrality(test_g)
                    end_py = time.time()
                    py_time = end_py - start_py
                    
                    # 尝试强制使用C++实现
                    os.environ['EASYGRAPH_CPP_ACCELERATOR'] = 'ON'
                    
                    start_cpp = time.time()
                    cpp_result = eg.eigenvector_centrality(test_g)
                    end_cpp = time.time()
                    cpp_time = end_cpp - start_cpp
                    
                    # 恢复原始环境变量
                    if old_env is None:
                        del os.environ['EASYGRAPH_CPP_ACCELERATOR']
                    else:
                        os.environ['EASYGRAPH_CPP_ACCELERATOR'] = old_env
                    
                    print(f"Python实现执行时间: {py_time:.6f}秒")
                    print(f"C++实现执行时间: {cpp_time:.6f}秒")
                    
                    if cpp_time < py_time:
                        print("C++实现似乎正常工作（执行更快）")
                    else:
                        print("警告：C++实现可能未正确加载（未显示性能优势）")
                except Exception as e:
                    print(f"测试C++/Python性能差异时出错: {e}")
            except Exception as e:
                print(f"调用失败，错误: {str(e)}")
                print("这表明C++实现可能存在问题")
        else:
            print("EasyGraph eigenvector_centrality 函数未配置为使用C++实现")
    except Exception as e:
        print(f"检查实现时出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        
        # 特别处理路径问题
        if "hypergraph" in str(e):
            print("\n检测到模块路径错误。正确的导入应为 easygraph.functions.centrality")
            try:
                from easygraph.functions.centrality import eigenvector_centrality
                print("成功从正确路径导入 eigenvector_centrality")
                source = inspect.getsource(eigenvector_centrality)
                cpp_call_present = "@hybrid(" in source
                print(f"是否配置了C++实现: {'是' if cpp_call_present else '否'}")
            except Exception as inner_e:
                print(f"尝试正确导入时出错: {inner_e}")

def measure_eigenvector_centrality(graphs, runs=3):
    """测量各库计算特征向量中心性的时间"""
    results = {}
    
    # 首先检查EasyGraph实现
    print("\n检查EasyGraph特征向量中心性实现:")
    check_eg_implementation()
    print("\n")
    
    # NetworkX
    nx_times = []
    for _ in range(runs):
        start_time = time.time()
        _ = nx.eigenvector_centrality(graphs['networkx'], max_iter=100)
        nx_times.append(time.time() - start_time)
    results['networkx'] = np.mean(nx_times)
    
    # iGraph
    ig_times = []
    for _ in range(runs):
        start_time = time.time()
        _ = graphs['igraph'].eigenvector_centrality()
        ig_times.append(time.time() - start_time)
    results['igraph'] = np.mean(ig_times)
    
    # Easy Graph 
    eg_times = []
    eg_used_cpp = False  # 标记是否使用了C++实现
    
    # 检查是否可以直接访问底层实现函数
    try:
        import easygraph.functions.centrality
        eg_impl = easygraph.functions.centrality.eigenvector_centrality.__code__
        eg_used_cpp = hasattr(easygraph, 'cpp_easygraph') and '@hybrid' in inspect.getsource(easygraph.functions.centrality.eigenvector_centrality)
        print(f"EasyGraph 是否使用C++实现: {'是' if eg_used_cpp else '否'}")
    except Exception as e:
        print(f"无法确定EasyGraph实现类型: {str(e)}")
    
    for _ in range(runs):
        start_time = time.time()
        _ = eg.eigenvector_centrality(graphs['easy_graph'])
        eg_times.append(time.time() - start_time)
    results['easy_graph'] = np.mean(eg_times)
    results['easy_graph_cpp'] = eg_used_cpp  # 记录是否使用了C++实现
    
    return results

def generate_datasets():
    """生成用于测试的图数据集"""
    datasets = []
    
    # 1. Zachary's Karate Club
    G_karate = nx.karate_club_graph()
    datasets.append(("karate_club", G_karate))
    
    # 2. Davis Southern Women Graph
    G_davis = nx.davis_southern_women_graph()
    datasets.append(("davis_southern_women", G_davis))
    
    # 3. Florentine Families Graph
    G_florentine = nx.florentine_families_graph()
    datasets.append(("florentine_families", G_florentine))
    
    # 4. 随机小型图
    G_erdos_small = nx.erdos_renyi_graph(1000, 0.1)
    datasets.append(("random_small", G_erdos_small))
    
    # 5. 随机中型图
    G_erdos_medium = nx.erdos_renyi_graph(5000, 0.05)
    datasets.append(("random_medium", G_erdos_medium))
    
    # 6. 随机大型图 (稀疏)
    G_erdos_large = nx.erdos_renyi_graph(10000, 0.01)
    datasets.append(("random_large_sparse", G_erdos_large))
    
    # 7. 小世界网络
    G_small_world = nx.watts_strogatz_graph(5000, 10, 0.1)
    datasets.append(("small_world", G_small_world))
    
    # 8. 无标度网络
    G_barabasi = nx.barabasi_albert_graph(5000, 5)
    datasets.append(("barabasi_albert", G_barabasi))
    
    return datasets

def main():
    # 生成测试数据集
    datasets = generate_datasets()
    
    results = []
    for name, graph in datasets:
        print(f"正在处理数据集: {name}")
        print(f"  节点数: {graph.number_of_nodes()}")
        print(f"  边数: {graph.number_of_edges()}")
        
        try:
            # 转换为各库的图格式
            graphs = convert_to_graphs(graph)
            
            # 测量算法性能
            timing_results = measure_eigenvector_centrality(graphs)
            
            # 添加到结果列表
            results.append({
                'dataset': name,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                **timing_results,
                'eg_used_cpp': timing_results.get('easy_graph_cpp', False)
            })
        except Exception as e:
            print(f"  处理数据集 {name} 时出错: {str(e)}")
    
    if not results:
        print("没有成功处理任何数据集")
        return
    
    # 创建比较表格
    df = pd.DataFrame(results)
    print("\n特征向量中心性算法性能比较 (时间单位: 秒):")
    print(df)
    
    # 保存结果
    df.to_csv('/users/sama/eigenvector_centrality_comparison.csv', index=False)
    
    # 打印C++实现状态
    cpp_status = all(r.get('eg_used_cpp', False) for r in results)
    print(f"\nEasyGraph特征向量中心性是否使用C++实现: {'是' if cpp_status else '否'}")
    
    if not cpp_status:
        print("\n诊断信息:")
        print("1. 检查C++文件路径是否正确 (cpp_easygraph/functions/centrality/)")
        print("2. 检查函数名称是否匹配 (cpp_eigenvector_centrality)")
        print("3. 检查编译是否成功")
        print("4. 检查Python代码中的hybrid装饰器是否正确配置")
        
        # 修正C++路径检测
        cpp_path = "/users/sama/Easy-Graph/cpp_easygraph/functions/centrality"
        print(f"\nEasyGraph C++路径: {cpp_path}")
        if os.path.exists(cpp_path):
            print(f"C++函数目录存在: {cpp_path}")
            print("文件列表:")
            for file in os.listdir(cpp_path):
                print(f"  - {file}")
                
            # 检查eigenvector.cpp文件
            eigenvector_cpp = os.path.join(cpp_path, "eigenvector.cpp")
            if os.path.exists(eigenvector_cpp):
                print(f"\n{eigenvector_cpp} 文件存在")
                # 可以进一步分析文件内容，例如检查函数声明是否正确
            else:
                print(f"\n{eigenvector_cpp} 文件不存在，这可能是问题所在")
        else:
            print(f"C++函数目录不存在: {cpp_path}")
            print("这是C++实现未能加载的主要原因")
    
    # 可视化比较结果
    plt.figure(figsize=(12, 6))
    
    # 只比较这三个库
    libraries = ['networkx', 'igraph', 'easy_graph']
    
    bar_width = 0.25  # 由于减少了一个库，增加每个条形的宽度
    index = np.arange(len(results))
    
    for i, lib in enumerate(libraries):
        times = [r[lib] if lib in r else 0 for r in results]
        plt.bar(index + i*bar_width, times, bar_width, label=lib)
    
    plt.xlabel('数据集')
    plt.ylabel('运行时间 (秒)')
    plt.title('特征向量中心性算法性能比较')
    plt.xticks(index + bar_width, [r['dataset'] for r in results], rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/users/sama/eigenvector_centrality_comparison.png')
    
    print("结果已保存到 /users/sama/eigenvector_centrality_comparison.csv")
    print("可视化结果已保存到 /users/sama/eigenvector_centrality_comparison.png")
    
    # 打印性能比较摘要
    print("\n性能比较摘要:")
    for lib in libraries:
        lib_results = [r[lib] for r in results if lib in r]
        if lib_results:
            avg_time = np.mean(lib_results)
            print(f"{lib}: 平均运行时间 {avg_time:.6f}秒")
    
    # 找出性能最好的库
    lib_avg_times = {lib: np.mean([r[lib] for r in results if lib in r]) 
                     for lib in libraries if any(lib in r for r in results)}
    if lib_avg_times:
        fastest_lib = min(lib_avg_times.items(), key=lambda x: x[1])[0]
        print(f"\n在测试的数据集上，{fastest_lib}平均性能最佳")
    
    plt.show()


if __name__ == '__main__':
    main()
