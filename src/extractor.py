import argparse

import xlsxwriter
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from config import Config, Promts, LogConfig
import logging

from logging.config import dictConfig


class ESGInfoExtractor:
    """A class for extracting ESG-related information from documents."""

    def __init__(self, input_path: str) -> None:
        """
        Initialize the ESGInfoExtractor with the given input path.

        :param input_path: Path to the input document.
        """

        logging_config_dict = LogConfig.__dict__
        logging.config.dictConfig(logging_config_dict)
        self.logger = logging.getLogger("extractor")
        self.input_data_path = input_path
        self.config = Config()
        self.promts = Promts()
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.config.api_key)
        self.llm = ChatOpenAI(
            model_name=self.config.openai_model_name,
            temperature=self.config.open_ai_model_temp,
            openai_api_key=self.config.api_key,
        )
        self.questions_results = []
        esg_promt = PromptTemplate(
            template=self.promts.prompt_template, input_variables=["text"]
        )
        self.esg_chain = LLMChain(llm=self.llm, prompt=esg_promt)

        self.db = None
        self.esg_info = []
        self.retriever = None

    def _save_results(self) -> None:
        """Save the extracted questions and answers to an Excel file."""
        self.logger.info("Saving results")
        workbook = xlsxwriter.Workbook(self.config.output_file_name)
        worksheet = workbook.add_worksheet()

        i = 0
        for _, (query, answer) in enumerate(self.questions_results, start=1):
            worksheet.write(i, 0, query)
            worksheet.write(i, 1, answer)
            i += 1

        worksheet.write(i, 0, self.config.other_data_excel_col_name)
        for i, answer in enumerate(self.esg_info, start=i+1):
            worksheet.write(i, 0, answer)
        workbook.close()

    def load_data(self) -> None:
        """Load data from the input document, split it, and initialize the FAISS database."""
        self.logger.info("Creating embeddings")
        loader = PyPDFLoader(self.input_data_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.splitter_chunk_size,
            chunk_overlap=self.config.splitter_chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        pages = loader.load_and_split(text_splitter=text_splitter)
        self.db = FAISS.from_documents(pages, self.embeddings)
        self.retriever = self.db.as_retriever()

    def extract(self) -> None:
        """Extract ESG-related questions and answers from the loaded data."""
        self.logger.info("Extracting info")
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)
        for title, query in self.promts.questions_promts:
            answer = qa_chain({"query": query})["result"]
            self.questions_results.append((title, answer))

        docs = self.db.similarity_search(
            self.promts.get_esg_promt, k=self.config.esg_num_of_docs
        )
        esg_results = "\n".join([p.page_content for p in docs])
        result = self.esg_chain.run(text=esg_results)

        self.esg_info = result.replace(".", ".\n").split("\n")

        self._save_results()


def main() -> None:
    """Parse the input file argument and starts the ESG information extraction process."""
    parser = argparse.ArgumentParser(description="ESG Info Extractor")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file")

    args = parser.parse_args()

    e = ESGInfoExtractor(args.input)
    e.load_data()
    e.extract()


if __name__ == "__main__":
    main()
