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

1. **Niigata_west_2m.npy** - This dataset includes the 2m height wind data used for training.
2. **Niigata_West_2m_WholeSDF.npy** - This dataset contains the whole SDF data used for training.
3. **Niigata_west_2m.npy** - This dataset includes the 2m height wind data used for training.
4. **Niigata_west_2m.npy** - This dataset includes the 2m height wind data used for training.
   

### Test Datasets
1. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
2. **Niigata_West_2m_UpDown_SDF.npy** - This dataset contains up-down sampled data for testing purposes.
3. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
4. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
5. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
6. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
7. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
8. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
9. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
10. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
11. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.
12. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.

## Usage

### 1. `trainscript.py`

The `trainscript.py` script is used to train a model. After training, the model will be saved to the specified path.

**Usage:**

```bash
python3 trainscript.py model_path train_data_path [train_sdf_data_path]
```

### 2. `trainscript.py`

The `trainscript.py` script is used to train a model. After training, the model will be saved to the specified path.

**Usage:**

```bash
python3 trainscript.py model_path train_data_path [train_sdf_data_path]
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
