from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from qiskit.quantum_info import PauliList
from HamiltonianGen import HoppingHam, PauliSymplectic, HubbardHam2D
from local_solver import localSolver # 确保你能导入原有的 localSolver
import numpy as np
import copy
import time

def extract_features(H, selected_qubits, active_matrix):
    """提取 6 维轻量级特征"""
    activities = active_matrix[:, selected_qubits].sum(axis=0)
    activities = np.sort(activities)[::-1] # 降序排列，消除顺序影响
    
    row_sums = active_matrix[:, selected_qubits].sum(axis=1)
    strong_couplings = np.sum(row_sums >= 2) # 共现度
    current_weight = active_matrix.sum()
    
    return np.array([
        activities[0], activities[1], activities[2], activities[3], activities[4], activities[5],
        strong_couplings, current_weight
    ])

def generate_training_data(qNum=30, localSize=6, searchDepth=3, num_restarts=10, steps_per_run=500):
    """
    生成训练数据。
    num_restarts: 重新初始化哈密顿量的次数（多收集前期好降的数据）
    steps_per_run: 每次初始化的迭代步数（避免收集太多后期的 0）
    """
    X_data = []
    y_data = []
    
    print(f"开始收集数据... 预计总样本数: {num_restarts * steps_per_run}")
    start_time = time.perf_counter()

    for run in range(num_restarts):
        rr = np.random.randint(20, qNum-1, 1)[0]
        H_initial_op = HoppingHam(rr, qNum)
        mapper = JordanWignerMapper()
        mapped_op = mapper.map(H_initial_op)
        
        initial_x_mat = np.fliplr(PauliList(mapped_op.paulis).x).astype(np.int32)
        initial_z_mat = np.fliplr(PauliList(mapped_op.paulis).z).astype(np.int32)
        num_paulis = len(mapped_op.paulis[0])

        print(f"--- 开启第 {run+1}/{num_restarts} 轮数据收集 ---")
        # 每次重启都拷贝一份干净的初始状态
        ps_global = PauliSymplectic(initial_x_mat.copy(), initial_z_mat.copy(), num_paulis)
        
        for step in range(steps_per_run):
            active_matrix = ps_global.Xrep | ps_global.Zrep
            qubit_weights = active_matrix.sum(axis=0) # 得到长度为 qNum 的一维数组
            p = qubit_weights + 1e-3
            p = p / p.sum()
            random_ints = np.random.choice(qNum, localSize, replace=False)
            
            # 【提取特征 X】 - 必须在操作前提取！
            features = extract_features(ps_global, random_ints, active_matrix)
            old_w = ps_global.weight()
            
            # 运行局部求解器
            ps_local = PauliSymplectic(ps_global.Xrep[:, random_ints], ps_global.Zrep[:, random_ints], localSize)
            ls = localSolver(ps_local, searchDepth=searchDepth)
            ls(show=False)
            
            # 更新全局状态
            ps_global.Xrep[:, random_ints] = ls.H.Xrep
            ps_global.Zrep[:, random_ints] = ls.H.Zrep
            
            # 【计算标签 y】 - 实际下降了多少
            new_w = ps_global.weight()
            drop_amount = max(0, old_w - new_w) 
            
            # 记录到列表
            X_data.append(features)
            y_data.append(drop_amount)
            
            if drop_amount > 0:
                 print(f"发现有效样本! 下降: {drop_amount}, 当前 Weight: {new_w}")

    # 将列表转换为 numpy 数组并保存
    X_array = np.array(X_data)
    y_array = np.array(y_data)
    
    np.savez('xgboost_training_data.npz', X=X_array, y=y_array)
    
    end_time = time.perf_counter()
    print(f"\n数据收集完成！耗时: {end_time - start_time:.2f} 秒.")
    print(f"数据已保存为 xgboost_training_data.npz. 包含 {len(X_array)} 条样本。")

if __name__ == "__main__":
    # 收集 20 次重启 * 每次 150 步 = 3000 个样本
    generate_training_data()
