# C-STS

## Data

The dataset for C-STS is stored in an encrypted file named `csts.tar.enc`. To access the dataset, follow these steps:

1. Request Access: Submit a request to obtain the decryption password by [clicking here](https://docs.google.com/forms/d/e/1FAIpQLSfoYig6I3qEBUBaNmzugnAKGpX1mSpM5cbGeO-dXq-u_sMPJQ/viewform?usp=sf_link). You will receive an email response with the password immediately.

2. Decrypt the Dataset: Once you have received the password via email, you can decrypt the `csts.tar.enc` file using the provided `extract.sh` script. Follow the instructions below:

   - Open a terminal and navigate to the directory where the file is located.
   - Run the following command, replacing `<password>` with the decryption password obtained via email:

     ```bash
     ./extract.sh csts.tar.enc <password>
     ```
    
    Provided the correct password, this step will generate three files `csts_train.csv`, `csts_validation.csv`, and `csts_test.csv`, the unencrypted dataset splits.


**Important: By using this dataset, you agree not to publicly share it's unencrypted contents.**
