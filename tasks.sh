echo "mWACH NLP Pipelines - Experiments"

conda init bash
echo "Activating conda environment..."
conda activate monk
echo "Finished activating environment"

#Experiment 9
#echo "Starting experiment 9"
#cd Experiment_09/
#python system_ft_mBERT.py
#cd ..
#echo "Experiment 9 completed successfully"

#Experiment 10
#echo "Starting experiment 10"
#cd Experiment_10/
#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/system_pretrain_train_lbl_v5.csv \
#    --validation_file input/system_pretrain_dev_lbl_v5.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 512 \
#    --overwrite_output_dir \
#    --report_to all \
#    --output_dir model

#python system_ft_mBERT.py
#cd ..
#echo "Experiment 10 completed successfully"

#echo "Starting experiment 11"
#cd Experiment_11/

#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/pretrain_train_pilot_v4.csv \
#    --validation_file input/pretrain_validation_pilot_v4.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 128 \
#    --overwrite_output_dir \
#    --report_to all \
#    --output_dir model

#python system_ft_mBERT.py

#cd ..
#echo "Experiment 11 completed successfully"

#echo "StStarting experiment "
#cd Experiment_12

#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/system_pretrain_train_v5.csv \
#    --validation_file input/system_pretrain_dev_v5.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 512 \
#    --overwrite_output_dir \
#    --report_to all \
#    --output_dir model

#python system_ft_mBERT.py

#cd ..
#echo "ExExperiment 12 completed successfully"


#echo "Starting experiment 13"
#cd Experiment_13


python run_mlm.py \
    --model_name_or_path "bert-base-multilingual-cased" \
    --train_file input/combined_pretrain_train_v5.csv \
    --validation_file input/combined_pretrain_dev_v5.csv \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --mlm_probability 0.15 \
    --num_train_epochs 5 \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --report_to all \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir model

#python system_ft_mBERT.py

#cd ..
#echo "Experiment 13 completed successfully"

echo "Starting experiment 14"
cd Experiment_14

python run_mlm.py \
    --model_name_or_path "bert-base-multilingual-cased" \
    --train_file input/system_pretrain_train_v6.csv \
    --validation_file input/system_pretrain_dev_v6.csv \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --mlm_probability 0.15 \
    --num_train_epochs 5 \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --report_to all \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir model

python system_ft_mBERT.py

#cd ..
#echo "Experiment 14 completed successfully"

#echo "Starting experiment 15"
#cd Experiment_15

python run_mlm.py \
    --model_name_or_path "bert-base-multilingual-cased" \
    --train_file input/system_pretrain_dev_lbl_v6.csv \
    --validation_file input/system_pretrain_train_lbl_v6.csv \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --mlm_probability 0.15 \
    --num_train_epochs 5 \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --report_to all \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --output_dir model
#
#python system_ft_mBERT.py
#
#cd ..
#echo "Experiment 15 completed successfully"
#
#
#echo "Starting experiment 18"
#cd Experiment_18
#
#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/combined_pretrain_train_lbl_v5.csv \
#    --validation_file input/combined_pretrain_dev_lbl_v5.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 512 \
#    --overwrite_output_dir \
#    --report_to all \
#    --output_dir model
#
#python system_ft_mBERT.py
#
#cd ..
#echo "Experiment 18 completed successfully"
#
#

#Experiment 23
#echo "Starting experiment 23"
#cd Experiment_23/
#python ft_mBERT.py
#cd ..
#echo "Experiment 23 completed successfully"

#Experiment 24
#echo "Starting experiment 24"
#cd Experiment_24/
#python ft_mBERT.py
#cd ..
#echo "Experiment 24 completed successfully"

#Experiment 25
#echo "Starting experiment 25"
#cd Experiment_25/
#python ft_mBERT.py
#cd ..
#echo "Experiment 25 completed successfully"

#echo "Starting experiment 26"
#cd Experiment_26

#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/system_train_df_v7.csv \
#    --validation_file input/system_dev_df_v7.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 512 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --overwrite_output_dir \
#    --report_to all \
#    --output_dir model

#python ft_mBERT.py

#cd ..
#echo "Experiment 26 completed successfully"

#echo "Starting experiment 27"
#cd Experiment_27

#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/nurse_train_df_v7.csv \
#    --validation_file input/nurse_dev_df_v7.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 512 \
#    --overwrite_output_dir \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 2 \
#    --report_to all \
#    --output_dir model

#python ft_mBERT.py

#cd ..
#echo "Experiment 27 completed successfully"

#echo "Starting experiment 28"
#cd Experiment_28

#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/system_pretain_train_v7.csv \
#    --validation_file input/system_pretrain_validation_v7.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 512 \
#    --overwrite_output_dir \
#    --report_to all \
#    --output_dir model

#python ft_mBERT.py

#cd ..
#echo "Experiment 28 completed successfully"


#echo "Starting experiment 29"
#cd Experiment_29

#python run_mlm.py \
#    --model_name_or_path "bert-base-multilingual-cased" \
#    --train_file input/nurse_pretain_train_v7.csv \
#    --validation_file input/nurse_pretrain_validation_v7.csv \
#    --do_train \
#    --do_eval \
#    --save_total_limit 1 \
#    --mlm_probability 0.15 \
#    --num_train_epochs 5 \
#    --max_seq_length 512 \
#    --overwrite_output_dir \
#    --report_to all \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --output_dir model

#python ft_mBERT.py

#cd ..
#echo "Experiment 29 completed successfully"

conda deactivate



