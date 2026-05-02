====================================================================================================
saved_time: 2026-05-01 11:52:07
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0007998108_adam0p0021399817_decay317_trial007 --optimizer aux-sign-auto-cos-inc --lrate 0.0021399816882859213 --lrate_decay 317 --muon_lrate 0.0007998108260764364 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0007998108_adam0p0021399817_decay317_trial007/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0007998108_adam0p0021399817_decay317_trial007
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 27 min
current_train_loss: 0.0023788940
current_train_psnr: 32.865799
testset_mean_loss: 0.0013680941
testset_mean_psnr: 28.790859
testset_mean_ssim: 0.962840
testset_mean_lpips: 0.029036
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009439671, psnr=30.250431, ssim=0.967894, lpips=0.024483
  image_001: loss=0.0014972743, psnr=28.246986, ssim=0.961690, lpips=0.030167
  image_002: loss=0.0017241343, psnr=27.634289, ssim=0.957262, lpips=0.032340
  image_003: loss=0.0018237500, psnr=27.390347, ssim=0.952946, lpips=0.035984
  image_004: loss=0.0018936748, psnr=27.226946, ssim=0.953774, lpips=0.036275
  image_005: loss=0.0013856929, psnr=28.583330, ssim=0.961521, lpips=0.044574
  image_006: loss=0.0013757480, psnr=28.614611, ssim=0.964707, lpips=0.026877
  image_007: loss=0.0009381503, psnr=30.277275, ssim=0.971844, lpips=0.021647
  image_008: loss=0.0014779252, psnr=28.303475, ssim=0.956668, lpips=0.032584
  image_009: loss=0.0017812129, psnr=27.492842, ssim=0.951391, lpips=0.034471
  image_010: loss=0.0010402288, psnr=29.828711, ssim=0.969973, lpips=0.021292
  image_011: loss=0.0009237397, psnr=30.344503, ssim=0.975213, lpips=0.017959
  image_012: loss=0.0009118250, psnr=30.400884, ssim=0.974289, lpips=0.019825
  image_013: loss=0.0008134598, psnr=30.896638, ssim=0.976311, lpips=0.018731
  image_014: loss=0.0012488683, psnr=29.034833, ssim=0.966713, lpips=0.023857
  image_015: loss=0.0012150720, psnr=29.153980, ssim=0.961885, lpips=0.027311
  image_016: loss=0.0012600005, psnr=28.996292, ssim=0.970110, lpips=0.020910
  image_017: loss=0.0011587266, psnr=29.360190, ssim=0.969447, lpips=0.024705
  image_018: loss=0.0013330990, psnr=28.751376, ssim=0.965910, lpips=0.025263
  image_019: loss=0.0013584718, psnr=28.669493, ssim=0.964106, lpips=0.026447
  image_020: loss=0.0016889373, psnr=27.723864, ssim=0.955525, lpips=0.046381
  image_021: loss=0.0009527474, psnr=30.210222, ssim=0.965049, lpips=0.029596
  image_022: loss=0.0015369705, psnr=28.133344, ssim=0.954600, lpips=0.033749
  image_023: loss=0.0022380694, psnr=26.501264, ssim=0.948731, lpips=0.040013
  image_024: loss=0.0016806076, psnr=27.745337, ssim=0.953443, lpips=0.030449
