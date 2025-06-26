with open('../../tasks/ecommerce_tasks.json', 'rb') as f:
    content = f.read()
    for i, b in enumerate(content):
        if b < 32 and b not in (9, 10, 13):  # 排除常规制表/换行
            print(f"Illegal byte at {i}: {b:#04x}")