| Egypt                          |       fps |      lpips |    psnr |       ssim | ckpt_path                                                                          |       fps_std |   lpips_std |   psnr_std |   ssim_std |   coarse_psnr |   coarse_psnr_std |   fine_lpips |   fine_lpips_std |   fine_psnr |   fine_psnr_std |   fine_ssim |   fine_ssim_std |   num_rays_per_sec |   num_rays_per_sec_std |
|:-------------------------------|----------:|-----------:|--------:|-----------:|:-----------------------------------------------------------------------------------|--------------:|------------:|-----------:|-----------:|--------------:|------------------:|-------------:|-----------------:|------------:|----------------:|------------:|----------------:|-------------------:|-----------------------:|
| vanilla-nerf:2023-08-01_015311 | 0.0244074 | nan        | 12.5865 | nan        | outputs/Egypt/vanilla-nerf/2023-08-01_015311/nerfstudio_models/step-000029999.ckpt |   0.000124289 |  nan        |    1.25218 | nan        |       12.7633 |           1.20794 |     0.967278 |        0.0739801 |     12.5865 |         1.25218 |    0.417619 |       0.0995492 |            12652.8 |                64.4315 |
| nerfacto:2023-07-18_225346     | 0.170045  |   0.35702  | 20.8356 |   0.638194 | outputs/Egypt/nerfacto/2023-07-18_225346/nerfstudio_models/step-000029999.ckpt     |   0.00918191  |    0.101257 |    2.37266 |   0.108798 |      nan      |         nan       |   nan        |      nan         |    nan      |       nan       |  nan        |     nan         |            88151.3 |              4759.9    |
| instant-ngp:2023-07-17_213553  | 0.179025  |   0.410446 | 19.9326 |   0.605525 | outputs/Egypt/instant-ngp/2023-07-17_213553/nerfstudio_models/step-000029999.ckpt  | nan           |  nan        |  nan       | nan        |      nan      |         nan       |   nan        |      nan         |    nan      |       nan       |  nan        |     nan         |            92806.5 |               nan      |