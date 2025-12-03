def get_names(file_path):
    names = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split(',')
                if parts:  # 确保分割后有字段
                    name = parts[0].strip()
                    if name:  # 防止空name
                        names.add(name)
    return names

# 读取两个文件的name集合
file1_names = get_names('./a100.txt')
file2_names = get_names('./c500.txt')

# 计算交集并输出
common_names = file1_names & file2_names
print("共同存在的name:", common_names)
