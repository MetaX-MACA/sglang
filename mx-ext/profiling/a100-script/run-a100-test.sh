
#mn="/home/sw/aliyun/regression_test/final-profilling-script/profill-final"
#mn="/home/sw/aliyun/regression_test/final-profilling-script/test-all/c500-script"
mn="/home/e24186/envirment-vllm/vllm_6.3_post1/prefill_decoding_datastatistic/c500-script/data"
#mn="/pde_share/ai_share/yjchen/profiler_data"
mn="./data"
declare -a MODEL_PATH

# source /home/m01149/miniconda3/bin/activate

for file in "$mn"/*.json; do
  
  if [ -f "$file" ]; then
   
    MODEL_PATH=$(basename "$file")
    echo "$MODEL_PATH" 
    MODEL_PATH+=("$MODEL_PATH")
  fi
done

for i in "${!MODEL_PATH[@]}"
do
  model_name="${MODEL_PATH[i]}"
  #log_file="mcblas-v4-gemm-$model_name.log"
  #logfile=$(basename "$log_file")
  echo "$model_name"
  tp=$mn/$model_name
  echo "$tp"
  echo "++++++++++++++++++"
  python3 analysis_profiler_json.py  $tp
  before_json=${model_name%%.json*}.txt
  echo "analysis_json.py--is ok !"
  echo "$before_json"
  tn=$mn/$before_json
  python3 static_profiler.py $tn
  echo "static_profiler.py--is ok !"
done

echo "all json is over !!!!"

#mn="/home/sw/aliyun/regression_test/final-profilling-script/profill-final"
mkdir result
cd result/
rm *.xlsx
cd ../
cp $mn/*.xlsx result/
