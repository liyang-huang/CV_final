#!/bin/bash

python3 main.py --input-left ./data/Synthetic/TL0.png --input-right ./data/Synthetic/TR0.png --output ./result/TL0_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL1.png --input-right ./data/Synthetic/TR1.png --output ./result/TL1_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL2.png --input-right ./data/Synthetic/TR2.png --output ./result/TL2_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL3.png --input-right ./data/Synthetic/TR3.png --output ./result/TL3_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL4.png --input-right ./data/Synthetic/TR4.png --output ./result/TL4_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL5.png --input-right ./data/Synthetic/TR5.png --output ./result/TL5_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL6.png --input-right ./data/Synthetic/TR6.png --output ./result/TL6_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL7.png --input-right ./data/Synthetic/TR7.png --output ./result/TL7_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL8.png --input-right ./data/Synthetic/TR8.png --output ./result/TL8_syn.pfm
python3 main.py --input-left ./data/Synthetic/TL9.png --input-right ./data/Synthetic/TR9.png --output ./result/TL9_syn.pfm

python3 main.py --input-left ./data/Real/TL0.bmp --input-right ./data/Real/TR0.bmp --output ./result/TL0_real.pfm
python3 main.py --input-left ./data/Real/TL1.bmp --input-right ./data/Real/TR1.bmp --output ./result/TL1_real.pfm
python3 main.py --input-left ./data/Real/TL2.bmp --input-right ./data/Real/TR2.bmp --output ./result/TL2_real.pfm
python3 main.py --input-left ./data/Real/TL3.bmp --input-right ./data/Real/TR3.bmp --output ./result/TL3_real.pfm
python3 main.py --input-left ./data/Real/TL4.bmp --input-right ./data/Real/TR4.bmp --output ./result/TL4_real.pfm
python3 main.py --input-left ./data/Real/TL5.bmp --input-right ./data/Real/TR5.bmp --output ./result/TL5_real.pfm
python3 main.py --input-left ./data/Real/TL6.bmp --input-right ./data/Real/TR6.bmp --output ./result/TL6_real.pfm
python3 main.py --input-left ./data/Real/TL7.bmp --input-right ./data/Real/TR7.bmp --output ./result/TL7_real.pfm
python3 main.py --input-left ./data/Real/TL8.bmp --input-right ./data/Real/TR8.bmp --output ./result/TL8_real.pfm
python3 main.py --input-left ./data/Real/TL9.bmp --input-right ./data/Real/TR9.bmp --output ./result/TL9_real.pfm
