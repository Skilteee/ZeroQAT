# need test seeds: 42, 26, 19, 58

# 4090上
# SST2 good use seed 26
# 11:03 (100 eval)
MODEL=../opt2p7b TASK=SST2 MODE=ft LR=1e-6 STEPS=2000 SEED=26 EPS=1e-3 bash mezo.sh
# 11:29 (100 eval, 100 projection, 0.3)
MODEL=../opt2p7b TASK=SST2 MODE=ft LR=1e-6 STEPS=2000 ENHANCED=zo SEED=26 EPS=1e-3 bash mezo.sh

MODEL=facebook/opt-2.7b TASK=SST2 MODE=lora LR=1e-4 STEPS=4000 SEED=42 EPS=1e-2 bash mezo.sh
# 08:54 (200 eval, 100 projection, 0.015)
MODEL=facebook/opt-2.7b TASK=SST2 MODE=lora LR=1e-4 STEPS=4000 SEED=42 ENHANCED=zo EPS=1e-2 bash mezo.sh

# RTE good use seed 42
# 1:23:44 (400 eval)
MODEL=../opt2p7b TASK=RTE MODE=ft LR=1e-6 STEPS=8000 SEED=42 EPS=1e-3 bash mezo.sh
# 1:30:32 (400 eval, 100 projection) 1:25:22 (400 eval, 400 projection, 0.3)
MODEL=../opt2p7b TASK=RTE MODE=ft LR=1e-6 STEPS=8000 SEED=42 ENHANCED=zo EPS=1e-3 bash mezo.sh
MODEL=facebook/opt-2.7b TASK=RTE MODE=lora LR=1e-4 STEPS=8000 SEED=42 EPS=1e-2 bash mezo.sh
MODEL=facebook/opt-2.7b TASK=RTE MODE=lora LR=1e-4 STEPS=8000 SEED=42 ENHANCED=zo EPS=1e-2 bash mezo.sh

# WIC
# 40:18 (400 eval)
MODEL=../opt2p7b TASK=WIC MODE=ft LR=1e-6 STEPS=8000 SEED=42 EPS=1e-3 bash mezo.sh

# SQuAD
MODEL=../opt2p7b TASK=SQuAD MODE=ft LR=1e-6 STEPS=6000 SEED=0 ENHANCED=zo EPS=1e-3 bash mezo.sh


# A6000上
# CB good
MODEL=../opt2p7b TASK=CB MODE=ft LR=1e-6 STEPS=8000 SEED=42 EPS=1e-3 bash mezo.sh
# best performance!!
MODEL=../opt2p7b TASK=CB MODE=ft LR=1e-6 STEPS=8000 SEED=3218 ENHANCED=zo EPS=1e-3 bash mezo.sh
MODEL=facebook/opt-2.7b TASK=CB MODE=lora LR=1e-4 STEPS=8000 SEED=42 EPS=1e-2 bash mezo.sh


# BoolQ
MODEL=../opt2p7b TASK=BoolQ MODE=ft LR=1e-6 STEPS=8000 SEED=42 EPS=1e-3 bash mezo.sh
# 8:59:36 (400 eval, 400 projection)
MODEL=../opt2p7b TASK=BoolQ MODE=ft LR=1e-6 STEPS=8000 SEED=42 ENHANCED=zo EPS=1e-3 bash mezo.sh
MODEL=facebook/opt-2.7b TASK=BoolQ MODE=lora LR=1e-4 STEPS=8000 SEED=42 ENHANCED=zo EPS=1e-2 bash mezo.sh


# WSC good
MODEL=../opt2p7b TASK=WSC MODE=ft LR=1e-6 STEPS=12000 SEED=42 EPS=1e-3 bash mezo.sh
# 2:20:44 (400 eval, 400 projection, 0.3)
MODEL=../opt2p7b TASK=WSC MODE=ft LR=1e-6 STEPS=12000 ENHANCED=zo SEED=42 EPS=1e-3 bash mezo.sh
MODEL=facebook/opt-2.7b TASK=WSC MODE=lora LR=1e-4 STEPS=12000 SEED=42 EPS=1e-2 bash mezo.sh

# WIC
MODEL=facebook/opt-2.7b TASK=WIC MODE=lora LR=1e-4 STEPS=8000 SEED=42 EPS=1e-2 bash mezo.sh

# MultiRC
# 2:34:34 (400 eval)
MODEL=../opt2p7b TASK=MultiRC MODE=ft LR=1e-6 STEPS=2800 SEED=42 EPS=1e-3 bash mezo.sh
#
MODEL=facebook/opt-2.7b TASK=MultiRC MODE=ft LR=1e-6 STEPS=2800 SEED=42 ENHANCED=zo EPS=1e-3 bash mezo.sh

# SQuAD
MODEL=../opt2p7b TASK=SQuAD MODE=ft LR=1e-6 STEPS=6000 SEED=0 EPS=1e-3 bash mezo.sh
MODEL=facebook/opt-2.7b TASK=SQuAD MODE=lora LR=1e-4 STEPS=6000 SEED=0 EPS=1e-2 bash mezo.sh

# DROP
MODEL=facebook/opt-2.7b TASK=DROP MODE=lora LR=1e-4 STEPS=4800 SEED=42 EPS=1e-2 bash mezo.sh


