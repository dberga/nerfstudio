export CUDA_VISIBLE_DEVICES=2,3;
ns-process-data video --data data/Museu-MAC/casco_ceramico/IMG_9158.MOV --output-dir data/Museu-MAC/casco_ceramico --no-gpu;
ns-train nerfacto --data data/Museu-MAC/casco_ceramico --vis wandb;
ns-process-data video --data data/Museu-MAC/jarrones_vidrio/IMG_9198.MOV --output-dir data/Museu-MAC/jarrones_vidrio --no-gpu;
ns-train nerfacto --data data/Museu-MAC/jarrones_vidrio --vis wandb;
