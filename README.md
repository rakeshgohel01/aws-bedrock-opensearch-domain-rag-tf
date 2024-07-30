# RAG using Lambda, S3, OpenSearch, and Streamlit with Amazon Bedrock Titan text embedding and Claude Sonnet 3.5

This repository provides sample code for using the Retrieval Augmented Generation (RAG) method with [Amazon Bedrock](https://aws.amazon.com/bedrock/) [Titan Embeddings Generation 1 (G1)](https://aws.amazon.com/bedrock/titan/) Large Language Model (LLM) to create text embeddings stored in [Amazon OpenSearch](https://aws.amazon.com/opensearch-service/) 

This solution utilizes an AWS Lambda function triggered by S3 to read data files and upload vector embeddings to OpenSearch. Additionally, a Streamlit application is provided to retrieve and display data stored in OpenSearch.

## Prerequisites

1. This was tested on Python 3.11/12
2. Install [OpenTofu](https://opentofu.org/docs/intro/install/) to create the OpenSearch cluster.
3. It is advised to work in a clean environment, using `virtualenv` or any other virtual environment manager.

    ```bash
    pip install virtualenv
    python -m virtualenv venv
    source ./venv/bin/activate
    ```

4. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

5. Go to the Model Access [page](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess) and enable the foundation models you want to use.

## Steps for using this sample code

### 1. Launch an OpenSearch cluster using Terraform

```bash
cd ./terraform
tofu init
tofu apply -auto-approve
```

> This cluster configuration is for testing purposes only, as its endpoint is public to simplify the use of this sample code.

### 2. Upload data to OpenSearch using AWS Lambda

The provided Lambda function reads JSON files from an S3 bucket, generates embeddings using Amazon Bedrock, and uploads the vectors to OpenSearch.

The format of JSON file is
["question1", "answer1"]
["question2", "answer2"]

1. Deploy the Lambda function using the provided SAM template:

    ```bash
    sam build
    sam deploy --guided
    ```

    > Provide the necessary parameters, such as `BucketName`, `IndexName`, and `Region`.

2. Upload your data file (e.g., `data1.json`) to the specified S3 bucket to trigger the Lambda function.

### 3. Query the LLM using RAG and the Streamlit application

The Streamlit application allows you to query the LLM and retrieve data stored in OpenSearch.

1. Start the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the URL provided by Streamlit to interact with the application.


## Cleanup

To clean up the resources created by Terraform:

```bash
cd ./terraform
tofu destroy # When prompted for confirmation, type 'yes' and press enter.
```

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This source is licensed under the MIT License. See the LICENSE file.
