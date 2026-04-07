import numpy as np
import xgboost as xgb

# 1. 加载刚才收集的数据
data = np.load('xgboost_training_data.npz')
X_train = data['X']
y_train = data['y']

print(f"训练集特征维度: {X_train.shape}, 标签维度: {y_train.shape}")

# 2. 定义轻量级 XGBoost 模型
model = xgb.XGBRegressor(
    n_estimators=500,       # 50 棵浅层树，保证推理速度
    max_depth=20,           
    learning_rate=0.1,
    n_jobs=-1
)

# 3. 训练模型 (眨眼间就能完成)
print("开始训练模型...")
model.fit(X_train, y_train)

# 4. 保存模型供主程序调用
model.save_model("light_surrogate.json")
print("模型已保存为 light_surrogate.json")
