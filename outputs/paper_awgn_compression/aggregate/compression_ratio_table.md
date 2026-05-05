# Compression Ratio Table

| semantic_channels | semantic bottleneck label | semantic_symbol_ratio | semantic_compression_ratio | train SNR policy | checkpoint |
| ---: | --- | ---: | ---: | --- | --- |
| 16 | high bottleneck compression | 0.333333 | 3.000000 | [0.0, 5.0, 10.0, 15.0, 20.0] | `outputs/train_cifar10_awgn/ts_05050150__ch_awgn__snr_0_5_10_15_20__sem_16__base_32__img_32__seed_42/best_model.pth` |
| 32 | medium bottleneck compression | 0.666667 | 1.500000 | [0.0, 5.0, 10.0, 15.0, 20.0] | `outputs/train_cifar10_awgn/ts_05050150__ch_awgn__snr_0_5_10_15_20__sem_32__base_32__img_32__seed_42/best_model.pth` |
| 64 | wide semantic bottleneck | 1.333333 | 0.750000 | [0.0, 5.0, 10.0, 15.0, 20.0] | `outputs/train_cifar10_awgn/ts_05050150__ch_awgn__snr_0_5_10_15_20__sem_64__base_32__img_32__seed_42/best_model.pth` |
