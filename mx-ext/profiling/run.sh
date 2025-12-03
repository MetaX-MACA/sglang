if [ $# -ne 1 ]; then
    echo Usage: "$0"' <analysis_case_name>'
    exit 1
fi

# source /home/m01149/miniconda3/bin/activate

rm *.xlsx
echo "=========\n Start analysis a100 data ============"
cd a100-script/
echo $PWD
bash ./run-a100-test.sh
cd -

echo "=========\n Start analysis C500 data ============"
cd c500-script/
echo $PWD
bash ./run-c500-test.sh
cd -


mkdir result
echo "=========\n Start mergedata data ============"

cd result

rm *.xlsx

cd ../

echo $1
python3 profiler_lhh.py $1
