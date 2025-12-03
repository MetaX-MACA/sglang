import pandas as pd
import sys


def convert_us_to_float(input_list):
    for i, item in enumerate(input_list):
        if ' us' in item:
            item_without_us = item.replace('us', '') 
            try:
                return float(item_without_us)
            except ValueError:
                print(f"Cannot convert '{item_without_us}  {i}' to float.")
                return None 
    return None 

def count_kernel(kernels):
  kernel_count_dict = {}
  for i, name in enumerate(kernels):
    if len(name) < 2:
      continue
    kernel_name = name.split(',')[0]
    time_consum = convert_us_to_float(name.split(','))

    if kernel_name not in kernel_count_dict.keys():
      count = 1
      kernel_count_dict[kernel_name] = [count, time_consum]
    else:
      count = kernel_count_dict[kernel_name][0]+1
      kernel_count_dict[kernel_name] = [count,kernel_count_dict[kernel_name][1] + time_consum]
  return kernel_count_dict

def to_dataframe(prefill_list, decoding_list, xlsx_dir):
    kernel_name_list = []
    count_list = []
    pre_time_sonsum_list = []
    dec_time_sonsum_list = []
    percentage_list = []
    time_sonsum_list = []
    stage_list = []
    ave_time_list = []
    
    prefill_total_time = 0.0
    decoding_total_time = 0.0
    # import pdb; pdb.set_trace()
    for pre in prefill_list:
      kernel_name_list.append(pre[0])
      count_list.append(pre[1][0])
      pre_time_sonsum_list.append(pre[1][1])
      prefill_total_time = prefill_total_time + pre[1][1]
      stage_list.append('prefill')
      ave_time_list.append(round(pre[1][1]/pre[1][0],2))
    
    #pre_kernel_per = [str(x*100/prefill_total_time)[:4] + '%' for x in pre_time_sonsum_list]
    pre_kernel_per = [str(round(float(x/prefill_total_time),4)*100)[:4] + '%' for x in pre_time_sonsum_list] #LIUHH
    #pre_kernel_per = [str(x*100/prefill_total_time) + '%' for x in pre_time_sonsum_list] #LIUHH

    total = 0.0
    for k in pre_time_sonsum_list:
      total = total + k/prefill_total_time
    print("Prefill_total_percent: {} %".format(round(total*100, 4)))
    #pdb.set_trace() #LIUHH
    kernel_name_list.append(' ')
    count_list.append(' ')
    time_sonsum_list.append(prefill_total_time)
    stage_list.append(' ')
    ave_time_list.append(' ')
    
    for de in decoding_list:
      kernel_name_list.append(de[0])
      count_list.append(de[1][0])
      dec_time_sonsum_list.append(de[1][1])
      decoding_total_time = decoding_total_time + de[1][1]
      stage_list.append('decoding')
      ave_time_list.append(round(de[1][1]/de[1][0],2))
      
    #dec_kernel_per = [str(x*100/decoding_total_time)[:4]+ '%' for x in dec_time_sonsum_list]
    dec_kernel_per = [str(round(float(x/decoding_total_time),4)*100)[:4]+ '%' for x in dec_time_sonsum_list] #LIUHH
    #dec_kernel_per = [str(x*100/decoding_total_time)+ '%' for x in dec_time_sonsum_list] #LIUHH

    total = 0.0
    for k in dec_time_sonsum_list:
      total = total + k/decoding_total_time
    print("Decode_total_percent: {} %".format(round(total*100, 4)))
    kernel_name_list.append(' ')
    count_list.append(' ')
    time_sonsum_list.append(decoding_total_time)
    stage_list.append(' ')
    ave_time_list.append(' ')
    
    kernel_df = pd.DataFrame()
    kernel_df['stage'] = stage_list
    kernel_df['Name'] = kernel_name_list
    kernel_df['Instances'] = count_list
    kernel_df['Total Time (us)'] = pre_time_sonsum_list + [prefill_total_time] + dec_time_sonsum_list + [decoding_total_time]
    kernel_df['Avg (us)'] = ave_time_list
    kernel_df['Time(%)'] = pre_kernel_per + [" "] + dec_kernel_per + [" "]
    
    kernel_df.to_excel(xlsx_dir,index=False)

def process_gemm(dict1):
    new_dict = {}
    gemm_count = 0
    gemm_time = 0
    gem_kernel_names = []
    for k, v in dict1.items():
        if 'gem' in k and 'grouped_gemm_' not in k:
            gemm_count = gemm_count + dict1[k][0]
            gemm_time = gemm_time + dict1[k][1]
            gem_kernel_names.append(k)
        else:
            new_dict[k] = v
    new_dict['gemm'] = [gemm_count, gemm_time]
    return new_dict, gem_kernel_names
    
def seg_prefill_decoding(file_path) :
    with open(file_path, 'r') as f:
        kernel_list = f.readlines()

    argmaxops_idxs = [i for i, j in enumerate(kernel_list) if 'ArgMaxOps' in j]
    seg_idxs = argmaxops_idxs[0::1] #索引1 : 第一个Token的结束即Prefill的结束，后面把所有的都分段处理

    name_list = []
    for kernel in kernel_list:
      name_list.append(kernel.strip().split(',')[0])
    
    seg_idx = 0
    first_idx = True
    prefill_count = 0
    decoding_count = 0
    for i in range(len(seg_idxs)):  #判断每个分段是Prefill Or Decode，最后一个Prefill分段为结束
      if first_idx:
          group_list = name_list[:(seg_idxs[i])]
          first_idx = False
      else:
          group_list = name_list[seg_idxs[i-1]:(seg_idxs[i])]

      set_list = list(set(group_list))
      if '_fwd_kerne' in set_list:
          prefill_count = prefill_count + 1
          seg_idx = seg_idxs[i]
      else:
          decoding_count = decoding_count + 1

    prefill_kernels = kernel_list[:seg_idx]
    decoding_kernels = kernel_list[seg_idx:]
    return prefill_kernels, decoding_kernels, prefill_count, decoding_count

def seg_prefill_decoding_new(file_path) :
    with open(file_path, 'r') as f:
        kernel_list = f.readlines()

    mark_idxs = [i for i, j in enumerate(kernel_list) if 'mark_' in j]
    seg_idxs = mark_idxs[0::1] #索引1 : 第一个Token的结束即Prefill的结束，后面把所有的都分段处理

    name_list = []
    for kernel in kernel_list:
      name_list.append(kernel.strip().split(',')[0])
    
    prefill_seg_idxs: List[Tuple[int, int]]  = []
    decode_seg_idxs: List[Tuple[int, int]]  = []
    first_idx = True
    prefill_count = 0
    decoding_count = 0
    next_stage_start_idx = 0
    prefill_kernels  = []
    decoding_kernels = []
    for i in range(len(seg_idxs)):  #判断每个分段是Prefill Or Decode，最后一个Prefill分段为结束
      group_list = name_list[next_stage_start_idx:(seg_idxs[i]+1)]

      set_list = list(set(group_list))
      if 'mark_prefill_kerne' in set_list:
          prefill_count = prefill_count + 1
          # prefill_seg_idxs.append((next_stage_start_idx, seg_idxs[i]))
          prefill_kernels.extend(kernel_list[next_stage_start_idx:seg_idxs[i]])
          print(f"prefill {next_stage_start_idx}, {seg_idxs[i]}")

      if 'mark_decode_kerne' in set_list:
          decoding_count = decoding_count + 1
          # decode_seg_idxs.append((next_stage_start_idx, seg_idxs[i]))
          decoding_kernels.extend(kernel_list[next_stage_start_idx:seg_idxs[i]])
          print(f"decode {next_stage_start_idx}, {seg_idxs[i]}")
      next_stage_start_idx = seg_idxs[i] + 1
      # print(next_stage_start_idx)

    return prefill_kernels, decoding_kernels, prefill_count, decoding_count

if __name__ == '__main__':
    file_path = sys.argv[1]
    print("------------   static_profiler start: {} ".format(file_path))
    prefill_kernels, decoding_kernels, prefill_count, decoding_count = seg_prefill_decoding_new(file_path)
    print("Prefill Lines: {}, Deode Lines: {}".format(len(prefill_kernels), len(decoding_kernels)))

    prefill_ = count_kernel(prefill_kernels)
    prefill_1, prefill_gem_names = process_gemm(prefill_)
    print("Prefill gemm: {} ".format(prefill_gem_names))
    new_prefill = sorted(prefill_1.items(), key=lambda d:d[1][1],reverse=True)
    print(" ")
    decoding_ = count_kernel(decoding_kernels)
    decoding_1, decoding_gem_names = process_gemm(decoding_)
    print("Decoding gemm: {} ".format(decoding_gem_names))
    new_decoding = sorted(decoding_1.items(), key=lambda d:d[1][1],reverse=True)

    out_dir = file_path.replace('txt', 'xlsx')
    to_dataframe(new_prefill, new_decoding, out_dir)
    print("------------   static_profiler end -----------------")
    # print('prefill-times :{}'.format(prefill_count))
    # print('decoding-times:{}'.format(decoding_count))