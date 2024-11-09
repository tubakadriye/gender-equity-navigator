import mesop as me
import asyncio
import logging
from llama_index.core import load_indices_from_storage
import os
from google.cloud import aiplatform
from google.auth import load_credentials_from_file
from llama_index.core import (
    StorageContext,
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

from typing import List, Optional
from llama_index.core.vector_stores import FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters
from pathlib import Path

from llama_index.core.agent import FunctionCallingAgent
from google.cloud import storage
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("This is a debug message")

API_KEY= "AIzaSyDeIRtW4T5liuHcz-i_Gj4lk7_k28iPEhU"
GOOGLE_API_KEY = API_KEY  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# Project and Storage Constants
PROJECT_ID = "gender-equity-navigator"
REGION = "europe-west1"
GCS_BUCKET_NAME = "gender-equity-research-docs"
GCS_BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
# The number of dimensions for the textembedding-gecko@003 is 768
# If other embedder is used, the dimensions would probably need to change.
VS_DIMENSIONS = 768
# Vertex AI Vector Search Index configuration
# parameter description here
# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.MatchingEngineIndex#google_cloud_aiplatform_MatchingEngineIndex_create_tree_ah_index
VS_INDEX_NAME = "gender_equity_vector_search_index"  # @param {type:"string"}
VS_INDEX_ENDPOINT_NAME = "gender_equity_vector_search_endpoint"  # @param {type:"string"}

# Create a global variable to track loading status
#indices_loaded = False
#indices = None  # Initially set to None until loading is complete

aiplatform.init(project=PROJECT_ID, location=REGION)
vs_index = aiplatform.MatchingEngineIndex(index_name="5918794237620518912")

vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name="6059426172859580416"
)
print(
        f"Vector Search index {vs_index.display_name} exists with resource name {vs_index.resource_name}"
    )
print(
        f"Vector Search index endpoint {vs_endpoint.display_name} exists with resource name {vs_endpoint.resource_name}"
    )


def check_gcs_connection(bucket_name):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs())
        print(f"Connected to GCS bucket '{bucket_name}'. Number of objects: {len(blobs)}")
        return True
    except Exception as e:
        print(f"Failed to connect to GCS bucket '{bucket_name}': {e}")
        return False

# Check GCS connection
gcs_connected = check_gcs_connection(GCS_BUCKET_NAME)

# Authenticate using the service account key file
credentials, project = load_credentials_from_file('gender-equity-navigator-5a54aed4da0b.json')
print(credentials)

# setup vector store
vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    index_id=vs_index.name,
    endpoint_id=vs_endpoint.name,
    gcs_bucket_name=GCS_BUCKET_NAME,
)

# set storage context
#storage_context = StorageContext.from_defaults(vector_store=vector_store)

#indices = load_indices_from_storage(storage_context)
#print(len(indices))
#if(len(indices)>0):
#    indices_loaded = True  # Set to True once loaded
#    logger.info("Indices loaded successfully.") 

vertex_gemini = Vertex(
    model="gemini-pro", temperature=1, additional_kwargs={}
)

# configure embedding model

embed_model = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project=PROJECT_ID,
    location=REGION,
    credentials = credentials
)

# setup the index/query process, ie the embedding model (and completion if used)
Settings.embed_model = embed_model
Settings.llm = vertex_gemini

# Run the async loading function at the start of the app
#asyncio.run(load_indices_async(storage_context))

index = VectorStoreIndex.from_vector_store(vector_store, embed_model )
print("Index loaded.")

# State class to manage the application state
@me.stateclass
class State:
    query: str = ""
    response: str = ""
    loading: bool = True  # Track loading status


@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io", "https://huggingface.co"]
    ),
    path="/",
    title="Gender Equity Navigator",
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ],
)

def page():
    try:
        #print("Rendering page")
        logger.debug("Rendering page")
        state = me.state(State)
        #state.loading = not indices_loaded  # Set loading status based on indices
        
        with me.box(
            style=me.Style(
                background="#4A306D",
                min_height="calc(100% - 48px)",
                padding=me.Padding(bottom=16),
            )
        ):
            #print("Rendering header text")
            logger.debug("Rendering header text")
            with me.box(
                style=me.Style(
                    width="min(970px, 100%)",
                    margin=me.Margin.symmetric(horizontal="auto"),
                    padding=me.Padding.symmetric(horizontal=16),
                )
            ):
                header_text()
                project_overview()
                upload_instructions()
                query_input()
                #if state.loading:
                #    me.text("Loading data, please wait...", style=me.Style(font_size=16, color="gray"))
                #else:
                if state.response:
                    display_response()  # Show responses only when not loading
            footer()
    except Exception as e:
        logger.error(f"Error in rendering page: {e}")



# Define UI components
def header_text():
    try:
        with me.box(
        style=me.Style(
            padding=me.Padding(top=64, bottom=36),
        )):
            me.text(
                "GENDER EQUITY NAVIGATOR",
                style=me.Style(
                    font_size=36,
                    font_weight=700,
                    text_align="center",
                    background="linear-gradient(90deg, #9B59B6, #FF80AB, #4C9F70) text",
                    position="sticky",
                    color="transparent",
                    margin=me.Margin(bottom=12)
                ),
            )
            me.text(
                "",
                style=me.Style(
                    font_size=20,
                    font_weight=700,
                    text_align="center",
                    position="sticky",
                    background="linear-gradient(90deg, #4285F4, #AA5CDB, #DB4437) text",
                    color="transparent",
                ),
            )
    except Exception as e:
        logger.error(f"Error in header_text: {e}")

def project_overview():
    with me.box(
        style=me.Style(
            padding=me.Padding.all(16),
            background="#F8F3FA", # Light pinkish background for a welcoming feel
            border_radius=16,
            margin=me.Margin(top=36),
            min_width="720px",
        )
    ):
        me.text(
            "Gender equity is a critical global challenge, and data-driven solutions can empower organizations to address it effectively. By leveraging Retrieval-Augmented Generation (RAG) technology, Gender Equity Navigator draws from a rich repository of global reports, articles, and research to provide clear, tailored insights and recommendations on gender equity challenges. With GEN, users can ask complex questions and receive detailed answers on topics like effective diversity policies, gender pay disparities, and leadership representation.",
            style=me.Style(
                font_size=16,
                line_height="1.5",
                color="#3A265A",  # Darker purple for text, symbolizing inclusivity and empowerment
                margin=me.Margin(bottom=16),
            ),
        )

        me.text(
            "With its accessible, data-centric approach, GEN aligns with UN SDG 5 (Gender Equality) and empowers users to make informed decisions that drive impactful, measurable progress toward gender equity.",
            style=me.Style(
                font_size=16,
                line_height="1.5",
                color="#3A265A",  # Darker purple for text, symbolizing inclusivity and empowerment
            ),
        )

def upload_instructions():
    with me.box(
        style=me.Style(
            padding=me.Padding.all(16),
            background="#F8F3FA",
            border_radius=16,
            margin=me.Margin(top=36),
            min_width="720px",
        )
    ):
        me.text(
            "Please enter your query about gender equity in the box below! GEN will analyze your query and provide tailored insights, reports, and actionable recommendations to support your goals in advancing gender equity.",
            style=me.Style(
                font_size=16,
                line_height="1.5",
                color="#3A265A",
            ),
        )


def footer():
    logger.debug("Rendering footer")
    with me.box(
        style=me.Style(
            padding=me.Padding.all(16),
            background="#F0F4F9",
            border_radius=16,
            margin=me.Margin(top=36, bottom=0),
            min_width="720px",
            text_align="center"
        )
    ):
        me.text(
            "© 2024 GenderEquityNavigator. All rights reserved.",
            style=me.Style(
                font_size=14,
                color="#888"
            )
        )

def query_input():
    state = me.state(State)
    with me.box(
        style=me.Style(
            border_radius=16,
            padding=me.Padding.all(16),
            background="white",
            display="flex",
            width="100%",
            box_shadow="0 4px 8px rgba(0, 0, 0, 0.1)",
            margin=me.Margin(top=24),
            min_width="720px",
        )
    ):
        with me.box(style=me.Style(flex_grow=1)):
            me.native_textarea(
                value=state.query,
                placeholder="Please enter your query about gender equity here",
                on_blur=on_blur,
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    outline="none",
                    width="100%",
                    border=me.Border.all(me.BorderSide(style="none")),
                ),
            )
        with me.content_button(type="icon", on_click=generate_response):
            me.icon("send")


def on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.query = e.value

def generate_response(e:me.ClickEvent):
    state = me.state(State)

    # Ensure indices are loaded before querying
    #if not indices_loaded:
    #    state.response = "Please wait, data is still loading..."
    #    return
    
    try:
        # Define query engines with loaded indices
        vector_query_engine = index.as_query_engine(
            similarity_top_k=10,
            streaming=True,
        )
        #summary_query_engine = indices[1].as_query_engine(
        #    response_mode="tree_summarize",
        #    use_async=True,
        #)

        # Define query tools
        #summary_tool = QueryEngineTool.from_defaults(
        #    query_engine=summary_query_engine,
        #    description="Useful for summarization questions related to Gender Equity over the years",
        #)
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description= ("Useful for retrieving specific context from Gender Equity articles and reports over the years."
                          ),
        )


        vector_query_tool = FunctionTool.from_defaults(
            name=f"vector_tool", fn=vector_query
        )
        vertex_gemini = Vertex(
            model="gemini-pro", temperature=1, additional_kwargs={}
        )

        # Create agent and query
        agent = FunctionCallingAgent.from_tools(
            [vector_tool], #summary_tool,
            llm=vertex_gemini,
            system_prompt="You are an agent designed to answer queries over a set of given articles and reports about gender equity.",
            verbose=True,
        )

        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                #summary_tool,
                vector_tool,
            ],
            verbose=True,
        )
        query = state.query
        print(query)

        #response = agent.query("What is the current status of the gender pay gap according to the Gender Snapshot reports?")
        response = agent.chat(query)
        state.response = str(response)
        logger.debug(f"Generated response for query: {state.query}")
        yield

    except IndexError as e:
        state.response = "Data is still loading, please try again shortly."
        logger.error(f"Index error: {e}")

    #yield  # Update the UI with the response

def vector_query(
        query: str, page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over the MetaGPT paper.

        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """

        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            filters=MetadataFilters.from_dicts(
                metadata_dicts, condition=FilterCondition.OR
            ),
        )
        response = query_engine.query(query)
        return response

def display_response():
    state = me.state(State)
    if state.response is not None:
        print(state.response)

        with me.box(
            style=me.Style(
                padding=me.Padding.all(16),  
                background="#F0F4F9",        
                border_radius=16,            
                margin=me.Margin(top=36),    
                box_shadow="0 4px 8px rgba(0, 0, 0, 0.1)",
                display="flex",              
                justify_content="center",   
                min_width="920px",         
                width="100%",               
                overflow_x="auto"           
            )
        ):
            with me.box(style=me.Style(
                min_width="720px",  
                width="100%",  
                overflow_x="auto"
            )):
                me.text(state.response,
                    style=me.Style(
                    font_size=16,
                    line_height="1.5",
                    ),
                )


