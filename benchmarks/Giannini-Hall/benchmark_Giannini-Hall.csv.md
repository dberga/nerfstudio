| experiment_name                | ckpt_path                                                                                  | fps                  | fps_std               | lpips              | lpips_std           | psnr               | psnr_std           | ssim               | ssim_std            | coarse_psnr       | coarse_psnr_std    | fine_lpips         | fine_lpips_std     | fine_psnr         | fine_psnr_std      | fine_ssim           | fine_ssim_std       | num_rays_per_sec | num_rays_per_sec_std |
| ------------------------------ | ------------------------------------------------------------------------------------------ | -------------------- | --------------------- | ------------------ | ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ----------------- | ------------------ | ------------------ | ------------------ | ----------------- | ------------------ | ------------------- | ------------------- | ---------------- | -------------------- |
| vanilla-nerf:2023-08-01_034801 | outputs/Giannini-Hall/vanilla-nerf/2023-08-01_034801/nerfstudio_models/step-000029999.ckpt | 0.008419201709330082 | 8.864850315148942e-06 |                    |                     | 9.710603713989258  | 0.6323637366294861 |                    |                     | 9.797874450683594 | 0.6915231347084045 | 1.0546433925628662 | 0.0449506938457489 | 9.710603713989258 | 0.6323637366294861 | 0.33742550015449524 | 0.08470986038446426 | 12662.48046875   | 13.332724571228027   |
| nerfacto:2023-07-17_232238     | outputs/Giannini-Hall/nerfacto/2023-07-17_232238/nerfstudio_models/step-000029999.ckpt     | 0.05460885539650917  |                       | 0.4717761278152466 |                     | 18.9600887298584   |                    | 0.5625701546669006 |                     |                   |                    |                    |                    |                   |                    |                     |                     | 82131.71875      |                      |
| nerfacto:2023-07-19_001806     | outputs/Giannini-Hall/nerfacto/2023-07-19_001806/nerfstudio_models/step-000029999.ckpt     | 0.05663774162530899  | 0.001365091884508729  | 0.4699304401874542 | 0.09604024887084961 | 18.976802825927734 | 2.156578540802002  | 0.5631924271583557 | 0.09345300495624542 |                   |                    |                    |                    |                   |                    |                     |                     | 85183.1640625    | 2053.09814453125     |