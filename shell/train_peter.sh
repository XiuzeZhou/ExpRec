echo $1
seed=1111
model_name=/PETER/
data_path=./data/
data_name=/reviews.pickle
#lr=1.0 0.5
checkpoint_dir=./models/
out_file_dir=./outputs/
out_file_name=generated.txt
out_log_dir=./logs/
out_log=train.log
n_layer=2

for d_type in TripAdvisor ClothingShoesAndJewelry MoviesAndTV Yelp  #
do
    for d_index in 1 2 3 4 5  # data_index
    do
        mkdir -p ${out_log_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${out_file_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${checkpoint_dir}${d_type}\/${d_index}${model_name}

        echo "data_type: $d_type, data_index: $d_index , seed: $seed, nlayers: $n_layer"
        TRANSFORMERS_CACHE=./llm/ \
        HF_DATASETS_CACHE=./llm/ \
        CUDA_VISIBLE_DEVICES=$1 D:/Anaconda3/envs/torch_2.2_rec/python -u ./train_peter.py \
            -data_path ${data_path}${d_type}${data_name} \
            -index_dir ${data_path}${d_type}\/${d_index}\/ \
            -emsize 512 \
            -nhead 2 \
            -nhid 2048 \
            -nlayers $n_layer \
            -dropout 0.2 \
            -lr 1.0 \
            -clip 1.0 \
            -epochs 100 \
            -batch_size 128 \
            -seed $seed \
            -cuda \
            -log_interval 200 \
            -checkpoint ${checkpoint_dir}${d_type}\/${d_index}${model_name} \
            -outf ${out_file_dir}${d_type}\/${d_index}${model_name}${out_file_name} \
            -vocab_size 20000 \
            -endure_times 5 \
            -rating_reg 0.1 \
            -context_reg 1.0 \
            -text_reg 1.0 \
            -words 15 \
            > ${out_log_dir}${d_type}\/${d_index}${model_name}${out_log}
    done
done
