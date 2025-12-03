from collections import defaultdict

# 创建两个字典来分别存储每个 kernel 的出现次数和总时间
kernel_counts = defaultdict(int)
kernel_times = defaultdict(float)

# 读取文件并处理每一行
with open('/home/m01149/torch_profile_analysis/a100-script/data/1740474746.2762177.trace.txt', 'r') as file:
    for line in file:
        # 分割每行，假设每行由 ',' 分隔
        parts = line.strip().split(',')

        if len(parts) >= 2:
            kernel_name = parts[0].strip()
            
            # 提取第二字段时间部分，去掉 " us" 并转换为浮动类型
            try:
                time_str = parts[1].strip()
                time = float(time_str.replace(' us', ''))
            except ValueError:
                # 如果第二个字段不是有效的时间值，跳过该行
                continue

            # 更新统计信息
            kernel_counts[kernel_name] += 1
            kernel_times[kernel_name] += time

# 将统计信息按 kernel 出现次数排序
sorted_kernel_stats = sorted(kernel_counts.items(), key=lambda item: item[1], reverse=True)

# 打印结果
total_count =0
for kernel, count in sorted_kernel_stats:
    total_time = kernel_times[kernel]
    total_count +=count
    print(f"{kernel}, {count}, {total_time:.2f}")
print(total_count)

