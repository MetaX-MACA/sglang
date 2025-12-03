import pandas as pd
import sys


def count_kernel(kernels):
  kernel_count_dict = {}
  for name in kernels:
    kernel_name = name.split(',')[0]
    time_consum = float(name.split(',')[1].split(' ')[0])      
    # print('lllllllllllllllllllllllllllll',kernel_name, time_consum)
    
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
  mn=0.0
  for k in pre_time_sonsum_list:
      mn=mn+float(str(((k*100)/prefill_total_time))[:4])
  print(mn)
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

    for k, v in dict1.items():
        if 'gem' in k:
            gemm_count = gemm_count + dict1[k][0]
            gemm_time = gemm_time + dict1[k][1]

        else:
            new_dict[k] = v
    new_dict['gemm'] = [gemm_count, gemm_time]
    return new_dict
    
def find_prefill_end(txt_in):
    with open(txt_in, 'r') as f:
        kernel_list = f.readlines()

    ArgMaxOps_count = 0  
    prefill_end_idx = -1 
    kernel_names = []  
    kernel_times = []  

    for line_idx, kernel in enumerate(kernel_list):
        name, time, v = kernel.strip().split(',') 
        kernel_names.append(name)
        kernel_times.append(time.split()[0])
        if "ArgMaxOps" in name:
            ArgMaxOps_count += 1
            if ArgMaxOps_count == 2:
                prefill_end_idx = line_idx 
                break

    return prefill_end_idx, kernel_names, kernel_times

if __name__ == '__main__':
    txt_in = sys.argv[1]
    with open(txt_in, 'r') as f:
      kernel_list = f.readlines()
    # import pdb
    #pdb.set_trace()
    # softmax_idx = [i for i, j in enumerate(kernel_list) if 'SoftMaxFor' in j]
    # even_idx = softmax_idx[1::2]

    # token_group = []

    # name_list = []
    # dur_list = []
    # triton_kernel_even_idx = 0;
    prefill_count = 0
    decoding_count = 0
    
    # seg_idx = 0
    # line_idx = 0
    # for kernel in kernel_list:
    #   line_idx++
    #   kernel_name = kernel.strip().split(',')[0]
    #   if kernel_name == 'triton_':
    #      triton_kernel_even_idx++
    #   if triton_kernel_even_idx == 3:
    #      seg_idx = line_idx
    #   name_list.append(kernel_name)
    #   dur_list.append(kernel.strip().split(',')[1])
    

    # first_idx = True
    # for i in range(len(even_idx)):
    #   if first_idx:
    #       group_list = name_list[:even_idx[i]]
    #       first_idx = False
    #   else:
    #       group_list = name_list[even_idx[i-1]:even_idx[i]]


    #   set_list = list(set(group_list))
    #   if 'flash_fwd_kernel' in set_list or 'flash::flash_fwd_kernel' in set_list:
    #       prefill_count = prefill_count + 1
    #       seg_idx = even_idx[i]
    #   else:
    #       decoding_count = decoding_count + 1
    prefill_end_idx, triton_names, triton_times = find_prefill_end(txt_in)
    prefill_kernel_list  = kernel_list[:prefill_end_idx]
    decoding_kernel_list = kernel_list[prefill_end_idx:]

    print(len(prefill_kernel_list))
    print(len(decoding_kernel_list))
    # import pdb
    # pdb.set_trace() #LIUHH
    prefill_ = count_kernel(prefill_kernel_list)
    prefill_1 = process_gemm(prefill_)
    new_prefill = sorted(prefill_1.items(), key=lambda d:d[1][1],reverse=True)
   # print(new_prefill)
    decoding_ = count_kernel(decoding_kernel_list)
    decoding_1 = process_gemm(decoding_)
    new_decoding = sorted(decoding_1.items(), key=lambda d:d[1][1],reverse=True)
  #  print(new_decoding)

    #LIUHH
  #   out_dir = 'gemm.xlsx'
    out_dir = txt_in.replace('txt','xlsx')
    to_dataframe(new_prefill, new_decoding, out_dir)
    # print('prefill-times :{}'.format(prefill_count))
    # print('decoding-times:{}'.format(decoding_count))

