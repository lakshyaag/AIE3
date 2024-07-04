from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from financial_chat.schema import ReportParams

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                """You are a financial report assistant, helping users understand financial reports (10-K or 10-Q) filed with the SEC in the United States.
                The current year is 2024. The current month is July.
                Respond in the specified format.
                """
            ),
        ),
        ("human", "{input}"),
    ]
)
llm = ChatOpenAI(model="gpt-4o", temperature=0.5).with_structured_output(ReportParams)

chain = prompt | llm
