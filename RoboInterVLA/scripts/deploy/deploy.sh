conda activate ManipVLA
cd ./real_experiment/manipvla_real_v2_OnlyW

CUDA_VISIBLE_DEVICES=0 python ./scripts/deploy/super_deploy.py --saved_model_path path/to/checkpoints/steps_1500_pytorch_model.pt --port 10011 --unnorm_key pick_pen_train_balance


