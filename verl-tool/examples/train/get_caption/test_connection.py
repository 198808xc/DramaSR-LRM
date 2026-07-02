import swanlab
import random
import time

# 初始化 SwanLab（离线模式）
swanlab.init(
    project="local_only_demo",
)

# 模拟训练过程
for step in range(10):
    acc = random.random()
    loss = 1 - acc
    # 记录日志
    swanlab.log({"accuracy": acc, "loss": loss}, step=step)
    time.sleep(0.5)

print("训练完成！日志已保存到本地 .swanlab/ 目录下。")
