# Privacy-Preserving Loan Approval ML Model

## Project Overview
This project implements a **Privacy-Preserving Loan Approval System** using encrypted machine learning. It ensures user data security while predicting loan approvals using **CrypTen** and **Concrete ML** for encrypted computations.

## Features
- **Privacy-Preserving Machine Learning** using CrypTen (MPC) and Concrete ML (Homomorphic Encryption).
- **Flask Web Application** with JWT authentication.
- **SQLite Database** for securely storing user details and predictions.
- **Secure User Inputs & Predictions** using encryption.

## Technologies Used
- **Python**
- **Flask** (Backend)
- **SQLite** (Database)
- **CrypTen** (Encrypted Model Training)
- **Concrete ML** (Homomorphic Encryption)
- **JWT Authentication**

## Installation
```sh
# Clone the repository
git clone https://github.com/VETHISARUN/privacy-preserving-loan-ml.git
cd privacy-preserving-loan-ml

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. **Start the Flask Server:**
   ```sh
   python app.py
   ```
2. **Access the Web App:** Open `http://localhost:5000` in your browser.
3. **Login & Predict:** Register, log in, and check loan eligibility securely.

## References
- [CrypTen](https://github.com/facebookresearch/CrypTen)
- [Concrete ML](https://www.zama.ai/concrete-ml)
- [Homomorphic Encryption](https://machinelearning.apple.com/research/homomorphic-encryption)

## License
This project is licensed under the MIT License.

