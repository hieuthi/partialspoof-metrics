# PartialSpoof Metrics

This repository focus on metrics for partial spoof tasks.

## Support metrics
- Utterance-based EER
- Frame-based EER
  - Support calculate result at higher or lower resolution than score unit
- Millisecond EER
  - Same idea as [Range-based EER](https://arxiv.org/abs/2305.17739) but simpler and faster implementation
- Accuracy, Precision, Recall, and F1

## Testing
Run the script in `examples/`` directory for testing.
```
==== Result Summary ====

Utterance EER: 1.48%
Utterance EER Threshold: threshold=0.8056 index=7014 accuracy=98.52% precision=99.83% recall=98.52% f1=99.17%
95% Recall Threshold: threshold=0.9264 index=7316 accuracy=95.50% precision=99.96% recall=95.02% f1=97.43%

Frame-based EER
0.02s   0.04s   0.08s   0.16s   0.32s   0.64s
13.72   14.46   15.29   11.60   9.63    7.24

Upscaled Utterance-based EER
0.02s   0.04s   0.08s   0.16s   0.32s   0.64s
1.55    1.50    1.55    1.66    1.90    2.24

Millisecond EER
0.02s   0.04s   0.08s   0.16s   0.32s   0.64s
14.62   16.29   18.27   20.31   26.27   34.54
```

## Contributions
Metrics are very important for research evaluation but can very tricky to implement.
If you find any bug or unsastifactory implementation or want to add new test case feel free to create a new topic in issue.

## Citation
Please cite this paper if you used this package for your research
```
@article{luong2025robust,
  title={Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation},
  author={Luong, Hieu-Thi and Rimon, Inbal and Permuter, Haim and Lee, Kong Aik and Chng, Eng Siong},
  journal={arXiv preprint arXiv:2507.03468},
  year={2025}
}

```

## License
[MIT License](LICENSE)
