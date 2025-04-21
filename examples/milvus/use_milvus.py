from pymilvus import Collection, utility

# 1. 连接 Milvus（v2.x+）
from pymilvus import connections
connections.connect(host="155.138.225.137", port="19530")

# 2. 获取集合对象
collection = Collection("vannasql")  # 替换为你的集合名
collection.load()  # 加载到内存

# 3. 查询数据（示例：获取前5条）
results = collection.query(
    expr="",  # 空字符串表示无过滤条件
    output_fields=["id", "embedding"],  # 要返回的字段
    limit=5
)

# 4. 打印结果
for item in results:
    print(f"ID: {item['id']}, Embedding (前5维): {item['embedding'][:5]}...")

# 5. 查看集合统计信息
print(f"行数: {collection.num_entities}")
# print(utility.describe_resource_group("firstLib"))