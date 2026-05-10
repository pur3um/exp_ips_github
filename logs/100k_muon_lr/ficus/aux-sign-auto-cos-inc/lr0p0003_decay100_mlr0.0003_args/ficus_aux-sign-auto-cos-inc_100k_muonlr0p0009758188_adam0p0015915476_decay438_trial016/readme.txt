====================================================================================================
saved_time: 2026-05-02 19:15:41
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0009758188_adam0p0015915476_decay438_trial016 --optimizer aux-sign-auto-cos-inc --lrate 0.0015915476453185764 --lrate_decay 438 --muon_lrate 0.0009758188097312541 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0009758188_adam0p0015915476_decay438_trial016/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0009758188_adam0p0015915476_decay438_trial016
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 32 min
current_train_loss: 0.0023917193
current_train_psnr: 32.276920
testset_mean_loss: 0.0013092148
testset_mean_psnr: 28.985967
testset_mean_ssim: 0.964436
testset_mean_lpips: 0.026191
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009060181, psnr=30.428631, ssim=0.968796, lpips=0.023158
  image_001: loss=0.0013773669, psnr=28.609503, ssim=0.963263, lpips=0.029019
  image_002: loss=0.0016382466, psnr=27.856207, ssim=0.959100, lpips=0.030324
  image_003: loss=0.0017868202, psnr=27.479191, ssim=0.954481, lpips=0.034619
  image_004: loss=0.0018510955, psnr=27.325712, ssim=0.954928, lpips=0.034663
  image_005: loss=0.0013033159, psnr=28.849503, ssim=0.964841, lpips=0.028492
  image_006: loss=0.0013020354, psnr=28.853772, ssim=0.965976, lpips=0.026846
  image_007: loss=0.0009104044, psnr=30.407656, ssim=0.972537, lpips=0.020217
  image_008: loss=0.0014788682, psnr=28.300705, ssim=0.958603, lpips=0.029755
  image_009: loss=0.0016063366, psnr=27.941634, ssim=0.954989, lpips=0.030202
  image_010: loss=0.0009900829, psnr=30.043284, ssim=0.971331, lpips=0.020419
  image_011: loss=0.0008646725, psnr=30.631483, ssim=0.976438, lpips=0.015968
  image_012: loss=0.0008652936, psnr=30.628365, ssim=0.975712, lpips=0.017472
  image_013: loss=0.0007865691, psnr=31.042630, ssim=0.977401, lpips=0.017201
  image_014: loss=0.0012408625, psnr=29.062763, ssim=0.966595, lpips=0.023591
  image_015: loss=0.0011615822, psnr=29.349500, ssim=0.962934, lpips=0.025127
  image_016: loss=0.0011858505, psnr=29.259700, ssim=0.971698, lpips=0.020559
  image_017: loss=0.0011046162, psnr=29.567885, ssim=0.971043, lpips=0.023256
  image_018: loss=0.0012434553, psnr=29.053698, ssim=0.968035, lpips=0.022574
  image_019: loss=0.0013574730, psnr=28.672688, ssim=0.964812, lpips=0.023263
  image_020: loss=0.0016300842, psnr=27.877899, ssim=0.957505, lpips=0.032262
  image_021: loss=0.0008891642, psnr=30.510180, ssim=0.967255, lpips=0.026716
  image_022: loss=0.0014616766, psnr=28.351487, ssim=0.956406, lpips=0.032322
  image_023: loss=0.0021711378, psnr=26.633126, ssim=0.950393, lpips=0.038184
  image_024: loss=0.0016173425, psnr=27.911980, ssim=0.955820, lpips=0.028567
