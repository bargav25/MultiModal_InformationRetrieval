conda activate rag_env

model_checkpoint="vidore/colqwen2-v1.0"

python task1_ColQwen2.py --model_checkpoint "$model_checkpoint" --start_idx 0 --end_idx 1658