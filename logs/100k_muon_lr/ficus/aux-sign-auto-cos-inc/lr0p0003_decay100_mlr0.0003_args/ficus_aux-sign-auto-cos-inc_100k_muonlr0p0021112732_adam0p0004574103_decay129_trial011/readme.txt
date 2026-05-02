====================================================================================================
saved_time: 2026-05-02 01:42:03
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0021112732_adam0p0004574103_decay129_trial011 --optimizer aux-sign-auto-cos-inc --lrate 0.00045741030960892004 --lrate_decay 129 --muon_lrate 0.0021112732276057943 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0021112732_adam0p0004574103_decay129_trial011/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0021112732_adam0p0004574103_decay129_trial011
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 26 min
current_train_loss: 0.0023996283
current_train_psnr: 32.232559
testset_mean_loss: 0.0012147184
testset_mean_psnr: 29.303371
testset_mean_ssim: 0.966947
testset_mean_lpips: 0.024097
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008674522, psnr=30.617544, ssim=0.970546, lpips=0.021119
  image_001: loss=0.0013258555, psnr=28.775038, ssim=0.965505, lpips=0.024892
  image_002: loss=0.0015704227, psnr=28.039834, ssim=0.961168, lpips=0.026581
  image_003: loss=0.0016748389, psnr=27.760269, ssim=0.957843, lpips=0.026536
  image_004: loss=0.0017457193, psnr=27.580256, ssim=0.959186, lpips=0.031076
  image_005: loss=0.0012338252, psnr=29.087463, ssim=0.967174, lpips=0.025439
  image_006: loss=0.0012125585, psnr=29.162973, ssim=0.967797, lpips=0.024469
  image_007: loss=0.0008270250, psnr=30.824813, ssim=0.975047, lpips=0.016713
  image_008: loss=0.0013003150, psnr=28.859514, ssim=0.961907, lpips=0.026468
  image_009: loss=0.0014096725, psnr=28.508817, ssim=0.958487, lpips=0.039793
  image_010: loss=0.0009203791, psnr=30.360332, ssim=0.973816, lpips=0.017004
  image_011: loss=0.0008128629, psnr=30.899826, ssim=0.978442, lpips=0.014107
  image_012: loss=0.0008296812, psnr=30.810887, ssim=0.977377, lpips=0.016042
  image_013: loss=0.0007396755, psnr=31.309587, ssim=0.979046, lpips=0.016290
  image_014: loss=0.0011579797, psnr=29.362990, ssim=0.969369, lpips=0.021581
  image_015: loss=0.0011169744, psnr=29.519567, ssim=0.965678, lpips=0.023384
  image_016: loss=0.0011098894, psnr=29.547203, ssim=0.973958, lpips=0.016767
  image_017: loss=0.0010750181, psnr=29.685842, ssim=0.972050, lpips=0.026945
  image_018: loss=0.0011010008, psnr=29.582123, ssim=0.970035, lpips=0.019543
  image_019: loss=0.0012389773, psnr=29.069366, ssim=0.967124, lpips=0.033413
  image_020: loss=0.0014626266, psnr=28.348665, ssim=0.961340, lpips=0.027865
  image_021: loss=0.0008192364, psnr=30.865907, ssim=0.970119, lpips=0.021128
  image_022: loss=0.0013612685, psnr=28.660562, ssim=0.958830, lpips=0.027610
  image_023: loss=0.0019781378, psnr=27.037434, ssim=0.953694, lpips=0.031240
  image_024: loss=0.0014765663, psnr=28.307470, ssim=0.958137, lpips=0.026424
