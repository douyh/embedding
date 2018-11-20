# embedding
some improvement on pair fast loss in pytorch

I reproduce pair fast loss in HDC with pytorch and make some improvement.

I use average pooling to get online confusion matrix and resample the train list.

I use resnet18 and single pair fast loss and reach 99.43% in 50sku.

Resampled list from online confusion matrix: 99.60%

Resampled list from offline confusion matrix: 99.64%


