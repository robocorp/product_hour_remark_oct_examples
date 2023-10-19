from ast import List
from typing import Any, Dict, Iterator
from robocorp.tasks import task
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessageChunk,
    BaseMessage,
)
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from robocorp.log import console_message
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)


load_dotenv()


@task
def run_chat():
    chat = ChatOpenAI()
    messages: List[BaseMessage] = [
        SystemMessage(content="You're a helpful assistant."),
        HumanMessage(content="Make a poem about a digital AI assistant called ReMark."),
    ]
    _write_stream_to_console(chat.stream(messages))


@task
def run_chat_and_see_response():
    chat = ChatOpenAI()
    messages: List[BaseMessage] = [
        SystemMessage(content="You're a helpful assistant."),
        HumanMessage(content="Make a poem about a digital AI assistant called ReMark."),
    ]
    result = chat.generate([messages])
    print(result)


@task
def run_chat_interactive():
    chat = ChatOpenAI()
    messages: List[BaseMessage] = [SystemMessage(content="You're a helpful assistant.")]
    while True:
        console_message("Human > ", "", flush=True)
        human_message = input()
        if not human_message.strip():
            break
        messages.append(HumanMessage(content=human_message))
        console_message("AI > ", "", flush=True)
        result = _write_stream_to_console(chat.stream(messages))
        messages.append(AIMessage(content=result))
    console_message("\n<END OF DISCUSSION>\n", "", flush=True)


@task
def load_document_embeddings_to_database():
    loader = UnstructuredFileLoader("./RobotFrameworkUserGuide.html", mode="elements")
    raw_documents = loader.load()
    for doc in raw_documents:
        doc.metadata = {"source": "Robot Framework User Guide"}
    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splitted_documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(
        splitted_documents, OpenAIEmbeddings(), persist_directory=".chroma"
    )
    # query it
    query = "What does CURDIR do?"
    data = db.similarity_search(query)

    # print results
    data[0].page_content


@task
def rag_bot():
    verbose = False
    db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=".chroma")
    llm = ChatOpenAI()
    combine_docs_chain = load_qa_chain(
        llm=llm,
        prompt=_answering_prompt(),
        verbose=verbose,
        document_prompt=_document_prompt(),
    )
    question_generator_chain = LLMChain(
        llm=llm, prompt=_data_query_generating_prompt(), verbose=verbose
    )
    chain = ConversationalRetrievalChain(
        combine_docs_chain=combine_docs_chain,
        retriever=db.as_retriever(),
        question_generator=question_generator_chain,
        verbose=verbose,
    )
    messages: List[BaseMessage] = []
    while True:
        console_message("Human > ", "", flush=True)
        human_message = input()
        if not human_message.strip():
            break
        console_message("AI > ", "", flush=True)
        result = chain({"question": human_message, "chat_history": messages})
        console_message(f"{result['answer']}\n", "", flush=True)
        messages.append(HumanMessage(content=human_message))
        messages.append(AIMessage(content=result["answer"]))
    console_message("\n<END OF DISCUSSION>\n", "", flush=True)


def _document_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """Content source: {source}\n{page_content}\n"""
    )


def _answering_prompt() -> PromptTemplate:
    system_template = """Use the following pieces of content to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
#Content
{context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    return ChatPromptTemplate.from_messages(messages)


def _data_query_generating_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """Combine the chat history and follow up question into
a standalone question. Chat History: {chat_history}
Follow up question: {question}"""
    )


def _write_stream_to_console(stream: Iterator[BaseMessageChunk]) -> str:
    full_message: List[str] = []
    for chunk in stream:
        console_message(chunk.content, "User messages", flush=True)
        full_message.append(chunk.content)
    console_message("\n", "User messages", flush=True)
    return "".join(full_message)
