# Generalization of urban wind field using Fourier Neural Operators (FNO) across different wind directions and cities

This repository contains the code necessary to reproduce the results presented in the paper "Generalization of urban wind field using Fourier Neural Operators (FNO) across different wind directions and cities". The paper explores innovative approaches and methodologies in the field of deep learning, providing valuable insights and advancements.

![West Output](pics/20240701TestPatches5mDelta0Niigata5in25outWestComparison0.gif)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Datasets

### Data link
The following datasets were used in this project:

[https://drive.google.com/drive/folders/1WDyBN6npsM2jiVIOl7QrFhZKiI6UJybV]


### Training Datasets

1. **NiigataTrain_256x256Whole_West_1020steps_5in10out.npy** - The training dataset for Niigata's west wind field, includes whole 256x256 data over 1020 steps.
2. **NiigataTrain_256x256Whole_West_1020steps_5in10out_SDF.npy** - SDF data corresponding to the Niigata whole 256x256 west wind field training dataset.
3. **NiigataTrain_64x64Patches_West_1020steps_5in10out.npy** - The training dataset for Niigata's west wind field, includes 64x64 patches over 1020 steps.
4. **NiigataTrain_64x64Patches_West_1020steps_5in10out_SDF.npy** - SDF data for the Niigata 64x64 patches west wind field training dataset.

### Test Datasets

1. **MontrealTest_64x64Patches_West_221steps.npy** - Test dataset for Montreal's west wind field using 64x64 patches over 221 steps.
2. **MontrealTest_64x64Patches_West_221steps_SDF.npy** - SDF data corresponding to the Montreal west wind field test dataset.
3. **NiigataTest_256x256Whole_NorthRotated90_1020steps.npy** - Test dataset for Niigata's north wind field, rotated 90 degrees, with whole 256x256 data over 1020 steps.
4. **NiigataTest_256x256Whole_NorthRotated90_1020steps_SDF.npy** - SDF data corresponding to the Niigata whole 256x256 north wind field test dataset (rotated 90 degrees).
5. **NiigataTest_64x64Patches_North_1020steps.npy** - Test dataset for Niigata's north wind field using 64x64 patches over 1020 steps.
6. **NiigataTest_64x64Patches_North_1020steps_SDF.npy** - SDF data for the Niigata north wind field test dataset using 64x64 patches.
7. **NiigataTest_64x64Patches_NorthRotated90_1020steps.npy** - Test dataset for Niigata's north wind field, rotated 90 degrees, using 64x64 patches over 1020 steps.
8. **NiigataTest_64x64Patches_NorthRotated90_1020steps_SDF.npy** - SDF data corresponding to the Niigata 64x64 patches north wind field test dataset (rotated 90 degrees).
9. **NiigataTest_64x64Patches_West_1020steps.npy** - Test dataset for Niigata's west wind field using 64x64 patches over 1020 steps.
10. **NiigataTest_64x64Patches_West_1020steps_SDF.npy** - SDF data for the Niigata west wind field test dataset using 64x64 patches.
11. **NiigataTest_64x64Patches_WestVerticallyFlipped_1020steps.npy** - Test dataset for Niigata's vertically flipped west wind field using 64x64 patches over 1020 steps.
12. **NiigataTest_64x64Patches_WestVerticallyFlipped_1020steps_SDF.npy** - SDF data for the Niigata vertically flipped west wind field test dataset using 64x64 patches.

## Usage

### 1. `Train_FNO.py`

The `trainscript.py` script is used to train a model. After training, the model will be saved to the specified path.

**Usage:**

```bash
python3 trainscript.py model_path train_data_path [train_sdf_data_path]
```

### 2. `Test_FNO.py`

The `testscript.py` script is used to test a pre-trained model. After testing, the predicted results and ground truth will be saved as `.npy` files.

**Usage:**

```bash
python3 testscript.py model_path train_data_path [train_sdf_data_path] test_data_path [test_sdf_data_path]
```



## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a Pull Request

7. ## References

In my work, I have referred to the following excellent projects and articles:

1. [Fourier Neural Operator](https://github.com/neuraloperator/neuraloperator) - The Fourier neural operator is the first ML-based method to successfully model turbulent flows with zero-shot super-resolution. It is up to three orders of magnitude faster compared to traditional PDE solvers. Additionally, it achieves superior accuracy compared to previous learning-based solvers under fixed resolution.
2. [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895) - Author: Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar


Special thanks to the authors of the work of FNO




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, please feel free to contact me at [j_chara@live.concordia.ca].
