from typing import ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Extractor configuration."""

    output_file_name: str = "results.xlsx"

    openai_model_name: str = "gpt-4"
    open_ai_model_temp: int = 0
    api_key: str = Field(alias="OPENAI_KEY")

    splitter_chunk_size: int = 1000
    splitter_chunk_overlap: int = 200

    esg_num_of_docs: int = 20

    other_data_excel_col_name: str = "Other sustainability data:"


class Promts(BaseSettings):
    """Extractor configuration."""

    questions_promts: list = [
        (
            "Article8",
            """
            Does company comply with the requirements of Article 8 under the 
            Sustainable Finance Disclosure Regulation (SFDR)? 
            Answer "Yes" or "No" 
         """,
        ),
        (
            "Article9",
            """
            Does your company comply with the requirements of Article 9 under the
            Sustainable Finance Disclosure Regulation (SFDR)? 
            Answer "Yes" or "No" 
            """,
        ),
    ]
    prompt_template: str = """
        Please very briefly summarize the main ideas of this text if they are relevant to ESG.
        No intoduction text, just the summary results.
        Text: {text}
        """
    get_esg_promt: str = "Any info about ESG (Environmental, Social, and Governance) "


class LogConfig(BaseModel):
    """Logging configuration to be set for the server."""

    LOGGER_NAME: str = "extractor"
    LOG_FORMAT: str = (
        "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    LOG_LEVEL: str = "INFO"
    version: ClassVar[int] = 1

    formatters: ClassVar[dict] = {
        "default": {
            "format": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: ClassVar[dict] = {
        "streamHandler": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    }

    loggers: ClassVar[dict] = {
        f"{LOGGER_NAME}": {
            "handlers": ["streamHandler"],
            "level": LOG_LEVEL,
        },
    }
