# C-STS

[[ArXiv]](https://arxiv.org/abs/2305.15093)

## Data


To avoid the intentional/unintentional scraping of the C-STS dataset that could impact the evaluation of LLM models trained from online data, we adopt the following approach for our dataset release.

The dataset for C-STS is stored in an encrypted file named `csts.tar.enc`. To access the dataset, follow these steps:

1. Request Access: Submit a request to obtain the decryption password by [clicking here](https://docs.google.com/forms/d/e/1FAIpQLSfoYig6I3qEBUBaNmzugnAKGpX1mSpM5cbGeO-dXq-u_sMPJQ/viewform?usp=sf_link). You will receive an email response with the password immediately.

2. Decrypt the Dataset: Once you have received the password via email, you can decrypt the `csts.tar.enc` file using the provided `extract.sh` script. Follow the instructions below:

   - Open a terminal and navigate to the directory where the `data` directory.
   - Run the following command, replacing `<password>` with the decryption password obtained via email:

     ```bash
     bash extract.sh csts.tar.enc <password>
     ```
    
    Provided the correct password, this step will generate three files `csts_train.csv`, `csts_validation.csv`, and `csts_test.csv`, the unencrypted dataset splits.

You can load the data using [datasets](https://github.com/huggingface/datasets) with the following lines

```python
from datasets import load_dataset

dataset = load_dataset(
  'csv', 
  data_files=
  {
    'train': 'data/csts_train.csv',
    'validation': 'data/csts_validation.csv',
    'test': 'data/csts_test.csv'
  }
)
```


**Important: By using this dataset, you agree to not publicly share its unencrypted contents or decryption password.**

## Citation
```tex
@misc{deshpande2023csts,
      title={CSTS: Conditional Semantic Textual Similarity}, 
      author={Ameet Deshpande and Carlos E. Jimenez and Howard Chen and Vishvak Murahari and Victoria Graf and Tanmay Rajpurohit and Ashwin Kalyan and Danqi Chen and Karthik Narasimhan},
      year={2023},
      eprint={2305.15093},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```