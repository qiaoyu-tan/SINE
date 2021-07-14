
# Three main parameters are needed to be tuned for various datasets.
# For small datasets, i.e., MovieLens, search alpha from {0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, intention k from {2, 4, 8}, and latent prototypes L from {10, 30, 50}
# For large datasets, i.e., Taobao, search alpha from {0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, intention k from {2, 4, 8, 10, 12, 16}, and latent prototypes L from {50, 100, 500, 1000, 2000, 5000}

python -u main.py --gpu 0 --dataset ml1m --topic_num 10 --category_num 2 --alpha 0.0 > sine_ml1m_tpc10_cat2_head1_alp0.0.txt
