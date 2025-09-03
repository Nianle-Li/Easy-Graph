# 保存为debug_cpp_graph.py
import easygraph as eg
import networkx as nx

def test_graph_structure():
    """分析Graph对象的结构"""
    G_nx = nx.path_graph(4)
    G_eg = eg.from_networkx(G_nx)
    
    print("Python Graph对象结构:")
    print(f"类型: {type(G_eg)}")
    print(f"属性: {dir(G_eg)}")
    
    # 尝试加载cpp_easygraph
    try:
        import cpp_easygraph
        print("\nC++模块中的函数:")
        for name in dir(cpp_easygraph):
            if not name.startswith("__"):
                print(f" - {name}")
                
        # 如果添加了诊断函数，可以调用它
        if hasattr(cpp_easygraph, "debug_graph_conversion"):
            print("\n执行图对象转换诊断:")
            cpp_easygraph.debug_graph_conversion(G_eg)
    except ImportError as e:
        print(f"无法导入C++模块: {e}")

if __name__ == "__main__":
    test_graph_structure()