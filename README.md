# PartialSpoof Metrics

This repository focus on metrics for partial spoof tasks.

## Support metrics
- Utterance-based EER
- Frame-based EER
- Millisecond EER
  - Same idea as [Range-based EER](https://arxiv.org/abs/2305.17739) but simpler and faster implementation
- Accuracy, Precision, Recall, and F1

## Testing
Run the script in `examples/`` directory for testing.
```
==== Result Summary ====

Utterance EER: 1.48%

Frame-based EER
0.02    0.04    0.08    0.16    0.32    0.64
13.72   14.46   15.29   11.60   9.63    7.24

Millisecond EER
0.02    0.04    0.08    0.16    0.32    0.64
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
