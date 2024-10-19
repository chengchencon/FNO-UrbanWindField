# Evaluation of FNO performance on 2D Urban Wind Field

This repository contains the code necessary to reproduce the results presented in the paper "Evaluation of FNO performance on 2D Urban Wind Field". The paper explores innovative approaches and methodologies in the field of [insert field of study, e.g., machine learning, data science, etc.], providing valuable insights and advancements.

![West Output](pics/20240701TestPatches5mDelta0Niigata5in25outWestComparison0.gif)

![West Train Error Vs Test Error](pics/accumulated_meanAbsoError_West_TrainVsTest.png)


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To get started with this project, you need to have Python installed. Follow the steps below to set up the environment and install the necessary libraries.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/chengchencon/FNO-Simulation.git
    cd FNO-Simulation
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required libraries**:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

After setting up the environment and installing the necessary libraries, you can run the script to get the results.

1. **Run the model script**:
    ```bash
    python3 FNO_Simulation_Train.py
    ```

2. **Check the output**:
    The results will be saved in the `results` directory. You can modify the script or use different input data as needed.

## Datasets

### Data link
The following datasets were used in this project:

https://drive.google.com/drive/folders/1QFJfiYjhq7O8VLob6MqcXUG-_9ajwDIK

### Test Datasets
1. **Niigata_West_2m_UpDown_SDF.npy** - This dataset contains up-down sampled data for testing purposes.
2. **Niigata_West_2m_UpDown.npy** - This dataset includes the up-down sampled data used for testing.

### Training Datasets
1. **Niigata_West_2m_WholeSDF.npy** - This dataset contains the whole SDF data used for training.
2. **Niigata_west_2m.npy** - This dataset includes the 2m height wind data used for training.


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
