export CUDA_VISIBLE_DEVICES=1;
#ns-process-data video --data data/Museu-DissenyHub/cobi_metal/IMG_9382.MOV --output-dir data/Museu-DissenyHub/cobi_metal;
#ns-train nerfacto --data data/Museu-DissenyHub/cobi_metal --vis wandb;
#ns-process-data video --data data/Museu-DissenyHub/bustos_porcelana/IMG_9409.MOV --output-dir data/Museu-DissenyHub/bustos_porcelana;
#ns-train nerfacto --data data/Museu-DissenyHub/bustos_porcelana --vis wandb;
ns-process-data video --data data/Museu-DissenyHub/maniquis_terciopelo/IMG_9442.MOV --output-dir data/Museu-DissenyHub/maniquis_terciopelo;
ns-train nerfacto --data data/Museu-DissenyHub/maniquis_terciopelo --vis wandb;
ns-process-data video --data data/Museu-DissenyHub/platos_laton/IMG_9392.MOV --output-dir data/Museu-DissenyHub/platos_laton;
ns-train nerfacto --data data/Museu-DissenyHub/platos_laton --vis wandb;
