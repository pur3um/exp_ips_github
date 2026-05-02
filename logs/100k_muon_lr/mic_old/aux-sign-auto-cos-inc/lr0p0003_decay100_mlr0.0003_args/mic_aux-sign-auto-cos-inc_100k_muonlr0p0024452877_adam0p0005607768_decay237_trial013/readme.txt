====================================================================================================
saved_time: 2026-05-01 23:08:39
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0024452877_adam0p0005607768_decay237_trial013 --optimizer aux-sign-auto-cos-inc --lrate 0.0005607767698867578 --lrate_decay 237 --muon_lrate 0.002445287720495803 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0024452877_adam0p0005607768_decay237_trial013/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0024452877_adam0p0005607768_decay237_trial013
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0013250737
current_train_psnr: 35.588802
testset_mean_loss: 0.0004595029
testset_mean_psnr: 33.594005
testset_mean_ssim: 0.980127
testset_mean_lpips: 0.022193
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004629479, psnr=33.344678, ssim=0.978759, lpips=0.014577
  image_001: loss=0.0002803960, psnr=35.522280, ssim=0.984941, lpips=0.018329
  image_002: loss=0.0005017251, psnr=32.995341, ssim=0.978863, lpips=0.024617
  image_003: loss=0.0004149607, psnr=33.819929, ssim=0.977750, lpips=0.026195
  image_004: loss=0.0004250060, psnr=33.716048, ssim=0.975450, lpips=0.026682
  image_005: loss=0.0003657567, psnr=34.368076, ssim=0.980165, lpips=0.026160
  image_006: loss=0.0003092806, psnr=35.096471, ssim=0.991488, lpips=0.010652
  image_007: loss=0.0004218601, psnr=33.748315, ssim=0.983320, lpips=0.013266
  image_008: loss=0.0005317336, psnr=32.743058, ssim=0.971362, lpips=0.037583
  image_009: loss=0.0004234291, psnr=33.732192, ssim=0.976095, lpips=0.021907
  image_010: loss=0.0002897553, psnr=35.379684, ssim=0.983034, lpips=0.015828
  image_011: loss=0.0002767355, psnr=35.579350, ssim=0.985709, lpips=0.017161
  image_012: loss=0.0011460490, psnr=29.407968, ssim=0.976702, lpips=0.022314
  image_013: loss=0.0006551985, psnr=31.836271, ssim=0.982243, lpips=0.016097
  image_014: loss=0.0005048052, psnr=32.968761, ssim=0.982930, lpips=0.016177
  image_015: loss=0.0004491144, psnr=33.476429, ssim=0.981096, lpips=0.017811
  image_016: loss=0.0003855985, psnr=34.138646, ssim=0.979132, lpips=0.017654
  image_017: loss=0.0004325734, psnr=33.639401, ssim=0.976409, lpips=0.027911
  image_018: loss=0.0003050615, psnr=35.156125, ssim=0.985312, lpips=0.013112
  image_019: loss=0.0004875989, psnr=33.119372, ssim=0.989213, lpips=0.011729
  image_020: loss=0.0005174278, psnr=32.861502, ssim=0.974741, lpips=0.037417
  image_021: loss=0.0005135375, psnr=32.894278, ssim=0.973634, lpips=0.039992
  image_022: loss=0.0005146367, psnr=32.884991, ssim=0.975320, lpips=0.031635
  image_023: loss=0.0005319482, psnr=32.741306, ssim=0.976704, lpips=0.028773
  image_024: loss=0.0003404351, psnr=34.679656, ssim=0.982811, lpips=0.021239
