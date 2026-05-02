====================================================================================================
saved_time: 2026-04-30 20:30:58
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0013274414_adam0p0010519726_decay269_trial001 --optimizer aux-sign-auto-cos-inc --lrate 0.0010519726230559959 --lrate_decay 269 --muon_lrate 0.0013274414292574908 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0013274414_adam0p0010519726_decay269_trial001/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0013274414_adam0p0010519726_decay269_trial001
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0009683491
current_train_psnr: 35.712723
testset_mean_loss: 0.0002331355
testset_mean_psnr: 37.129506
testset_mean_ssim: 0.980773
testset_mean_lpips: 0.016629
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001939747, psnr=37.122547, ssim=0.981811, lpips=0.015467
  image_001: loss=0.0001818899, psnr=37.401912, ssim=0.983439, lpips=0.016867
  image_002: loss=0.0002300091, psnr=36.382547, ssim=0.979371, lpips=0.019778
  image_003: loss=0.0001533851, psnr=38.142164, ssim=0.982482, lpips=0.018446
  image_004: loss=0.0001153997, psnr=39.377950, ssim=0.984887, lpips=0.014335
  image_005: loss=0.0001589510, psnr=37.987364, ssim=0.983562, lpips=0.012128
  image_006: loss=0.0002116737, psnr=36.743330, ssim=0.982663, lpips=0.013102
  image_007: loss=0.0001385015, psnr=38.585452, ssim=0.985206, lpips=0.012418
  image_008: loss=0.0001439885, psnr=38.416720, ssim=0.980833, lpips=0.012382
  image_009: loss=0.0001643074, psnr=37.843425, ssim=0.975094, lpips=0.021167
  image_010: loss=0.0001493279, psnr=38.258588, ssim=0.976375, lpips=0.022162
  image_011: loss=0.0002501636, psnr=36.017757, ssim=0.974708, lpips=0.022690
  image_012: loss=0.0002404873, psnr=36.189077, ssim=0.983680, lpips=0.011530
  image_013: loss=0.0001532958, psnr=38.144695, ssim=0.988926, lpips=0.008983
  image_014: loss=0.0002822398, psnr=35.493815, ssim=0.983277, lpips=0.016793
  image_015: loss=0.0012209186, psnr=29.133133, ssim=0.964678, lpips=0.032323
  image_016: loss=0.0005539417, psnr=32.565359, ssim=0.968359, lpips=0.028084
  image_017: loss=0.0001598455, psnr=37.962993, ssim=0.982580, lpips=0.015491
  image_018: loss=0.0002303707, psnr=36.375726, ssim=0.979855, lpips=0.012746
  image_019: loss=0.0001837474, psnr=37.357786, ssim=0.983394, lpips=0.012015
  image_020: loss=0.0001173852, psnr=39.303862, ssim=0.986332, lpips=0.011197
  image_021: loss=0.0001202001, psnr=39.200948, ssim=0.983198, lpips=0.011946
  image_022: loss=0.0001320666, psnr=38.792066, ssim=0.981378, lpips=0.019613
  image_023: loss=0.0001445321, psnr=38.400353, ssim=0.981922, lpips=0.018642
  image_024: loss=0.0001977845, psnr=37.038074, ssim=0.981306, lpips=0.015428
