from collections import defaultdict

# 创建一个字典来存储每个 kernel 的统计信息
kernel_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})

# 读取文件并处理每一行
with open('./output8.trace.txt', 'r') as file:
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
            
           # if kernel_name == 'cutlass::Kernel2':
               # print(kernel_name)
            # 更新统计信息
            kernel_stats[kernel_name]['count'] += 1
            kernel_stats[kernel_name]['total_time'] += time

# 将统计信息按 kernel 出现次数排序
sorted_kernel_stats = sorted(kernel_stats.items(), key=lambda item: item[1]['count'], reverse=True)

# 打印结果
print(len(sorted_kernel_stats))
for kernel, stats in kernel_stats.items():
    print(f"Kernel: {kernel}, Count: {stats['count']}, Total Time: {stats['total_time']:.2f} us")

