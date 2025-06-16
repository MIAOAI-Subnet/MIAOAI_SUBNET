import chardet

# 读取文件的前一部分字节进行编码推测
with open("/Users/zjp/code/python/cognify_github_miaoai/tasks/ecommerce_tasks.json", "rb") as f:
    raw_data = f.read(1024)  # 只读取前1024字节
    result = chardet.detect(raw_data)
    print(f"编码可能是: {result['encoding']}, 置信度: {result['confidence']}")

with open("/Users/zjp/code/python/cognify_github_miaoai/tasks/ecommerce_tasks.json", "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if "�" in line:  # � 是非法字符的替代符号
            print(f"第{i+1}行可能有非法字符: {line}")

import json

with open("/Users/zjp/code/python/cognify_github_miaoai/tasks/ecommerce_tasks.json", "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
        print("✅ JSON 文件解析成功")
    except json.JSONDecodeError as e:
        print("❌ JSON 解析失败:", e)