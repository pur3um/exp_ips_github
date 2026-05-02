====================================================================================================
saved_time: 2026-05-01 01:52:08
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial004 --optimizer aux-sign-auto-cos-inc --lrate 0.0011095983846308643 --lrate_decay 471 --muon_lrate 0.0003663671900803624 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial004/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial004
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0035138351
current_train_psnr: 29.724924
testset_mean_loss: 0.0016706102
testset_mean_psnr: 27.956288
testset_mean_ssim: 0.840900
testset_mean_lpips: 0.134321
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0014284290, psnr=28.451413, ssim=0.770452, lpips=0.198916
  image_001: loss=0.0016929321, psnr=27.713604, ssim=0.770586, lpips=0.202499
  image_002: loss=0.0014551160, psnr=28.371024, ssim=0.793043, lpips=0.177869
  image_003: loss=0.0017348449, psnr=27.607393, ssim=0.769695, lpips=0.194153
  image_004: loss=0.0014712544, psnr=28.323122, ssim=0.802089, lpips=0.149072
  image_005: loss=0.0013317495, psnr=28.755774, ssim=0.840111, lpips=0.129183
  image_006: loss=0.0017841361, psnr=27.485720, ssim=0.837808, lpips=0.135114
  image_007: loss=0.0016278175, psnr=27.883943, ssim=0.849865, lpips=0.120983
  image_008: loss=0.0013979683, psnr=28.545027, ssim=0.864711, lpips=0.111429
  image_009: loss=0.0014168173, psnr=28.486861, ssim=0.874240, lpips=0.102199
  image_010: loss=0.0012542093, psnr=29.016299, ssim=0.894783, lpips=0.081770
  image_011: loss=0.0024237738, psnr=26.155079, ssim=0.861491, lpips=0.110833
  image_012: loss=0.0033672950, psnr=24.727188, ssim=0.849581, lpips=0.135668
  image_013: loss=0.0028103904, psnr=25.512333, ssim=0.864151, lpips=0.113859
  image_014: loss=0.0022226092, psnr=26.531369, ssim=0.891933, lpips=0.093933
  image_015: loss=0.0014630315, psnr=28.347463, ssim=0.912514, lpips=0.076543
  image_016: loss=0.0013118944, psnr=28.821011, ssim=0.896106, lpips=0.077738
  image_017: loss=0.0015733655, psnr=28.031703, ssim=0.874841, lpips=0.099456
  image_018: loss=0.0021986759, psnr=26.578388, ssim=0.845195, lpips=0.133857
  image_019: loss=0.0017036824, psnr=27.686113, ssim=0.840822, lpips=0.130958
  image_020: loss=0.0014087528, psnr=28.511652, ssim=0.849018, lpips=0.132607
  image_021: loss=0.0013230018, psnr=28.784395, ssim=0.831547, lpips=0.140172
  image_022: loss=0.0010587476, psnr=29.752075, ssim=0.832285, lpips=0.148141
  image_023: loss=0.0010172168, psnr=29.925864, ssim=0.818960, lpips=0.171070
  image_024: loss=0.0012875431, psnr=28.902382, ssim=0.786683, lpips=0.190015
