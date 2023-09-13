# ESGInfoExtractor

## Description

The `ESGInfoExtractor` is a tool designed to extract Environmental, Social, and Governance (ESG) related information from documents. Using the power of `langchain` libraries and the OpenAI API, it reads data from PDFs, splits them into manageable chunks, and fetches ESG related questions and answers. Extracted information is then saved into an Excel file.

## Features

- Uses the OpenAI API for embeddings and language models.
- Supports PDF documents as input.
- Extracts and saves the ESG related information into an Excel file.

## Usage

1.  Run the script using Python:
```
poetry install
poetry shell
export OPENAI_KEY=<your key>
python extractor.py -i /path/to/your/input.pdf
```

The `-i` argument specifies the path to your input PDF document.

2. After the script finishes its execution, you'll find the extracted ESG data saved in an Excel file as specified in your `config`.

## Customization

You can customize the behavior of the extractor by modifying the `config.py` file. The configurations allow you to:

- Set the OpenAI API key.
- Define the OpenAI model's name and temperature.
- Specify the output Excel file name.
- Set other relevant configurations like the prompt template, splitter chunk size, etc.
