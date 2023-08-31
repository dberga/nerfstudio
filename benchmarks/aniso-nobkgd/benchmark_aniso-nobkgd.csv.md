| experiment_name            | ckpt_path                                                                             | fps                 | fps_std                | lpips               | lpips_std          | psnr               | psnr_std           | ssim               | ssim_std           | coarse_psnr        | coarse_psnr_std    | fine_lpips          | fine_lpips_std     | fine_psnr          | fine_psnr_std      | fine_ssim          | fine_ssim_std       | num_rays_per_sec | num_rays_per_sec_std |
| -------------------------- | ------------------------------------------------------------------------------------- | ------------------- | ---------------------- | ------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------- | ---------------- | -------------------- |
| mipnerf:2023-06-29_134624  | outputs/aniso-nobkgd/mipnerf/2023-06-29_134624/nerfstudio_models/step-000999999.ckpt  | 0.0222778283059597  | 0.00011478695523692295 |                     |                    | 16.110807418823242 | 2.0257012844085693 |                    |                    | 15.407804489135742 | 2.4511685371398926 | 0.39089006185531616 | 0.1444401741027832 | 16.110807418823242 | 2.0257012844085693 | 0.6965488195419312 | 0.16857467591762543 | 11548.826171875  | 59.50553512573242    |
| nerfacto:2023-07-07_173608 | outputs/aniso-nobkgd/nerfacto/2023-07-07_173608/nerfstudio_models/step-000029999.ckpt | 0.17043662071228027 | 0.005294472444802523   | 0.38229042291641235 | 0.2084013670682907 | 18.799728393554688 | 5.2370381355285645 | 0.6781659126281738 | 0.1589435636997223 |                    |                    |                     |                    |                    |                    |                    |                     | 88354.34375      | 2744.65478515625     |