import mesop as me
import logging
import os
from google.cloud import aiplatform
from google.auth import load_credentials_from_file
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from google.cloud import storage
import markdown
from vertexai.generative_models import (
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting
)


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
VS_INDEX_NAME = "gender_equity_vector_search_object_index"  # @param {type:"string"}
VS_INDEX_ENDPOINT_NAME = "gender_equity_vector_search_object_endpoint"  # @param {type:"string"}

aiplatform.init(project=PROJECT_ID, location=REGION)
vs_index = aiplatform.MatchingEngineIndex(index_name="7061582643065782272")

vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name="4695257698231386112"
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
credentials, project = load_credentials_from_file('./gender-equity-navigator-b38495299082.json')
print(credentials)

# setup vector store
vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    index_id=vs_index.name,
    endpoint_id=vs_endpoint.name,
    gcs_bucket_name=GCS_BUCKET_NAME,
)

vertex_gemini = Vertex(
    model="gemini-pro", temperature=1, additional_kwargs={}, 
    safety_settings = [SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        #method= generative_models.HarmBlockMethod.SEVERITY,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    )]
)

embed_model = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project=PROJECT_ID,
    location=REGION,
    credentials = credentials
)

# setup the index/query process, ie the embedding model (and completion if used)
Settings.embed_model = embed_model
Settings.llm = vertex_gemini

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
            margin=me.Margin(top=0, bottom= 0),
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

    vertex_gemini = Vertex(
    model="gemini-pro", temperature=1, additional_kwargs={}, 
    safety_settings = [SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        #method= generative_models.HarmBlockMethod.SEVERITY,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        #method= generative_models.HarmBlockMethod.SEVERITY,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        #method= generative_models.HarmBlockMethod.SEVERITY,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        #method= generative_models.HarmBlockMethod.SEVERITY,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    )
    ]
    )

    embed_model = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project=PROJECT_ID,
    location=REGION,
    credentials = credentials
    )
    
    try:
        # Define query engines with loaded indices
        vector_query_engine = index.as_query_engine(
            llm= vertex_gemini, 
            similarity_top_k=10,
            streaming=True,
            embed_model = embed_model
            
        )
        query = state.query
        print(state.query)

        response = vector_query_engine.query(query)
        #response = agent.query("What is the current status of the gender pay gap according to the Gender Snapshot reports?")
        #response = agent.chat(query)

        source_nodes = response.source_nodes
        response_data = {
            'Response Summary': response.response_txt,
            'Source Documents': []
        }

        # Loop through the source nodes and format the metadata and text content
        for node in source_nodes:
            metadata_str = "\n".join([f"{key}: {value}" for key, value in node.metadata.items()])
            # Use node.text for the content, and format it with the metadata string
            #formatted_text = f"Metadata:\n{metadata_str}\n\nContent:\n{node.text}"
            
            response_data['Source Documents'].append({
                'Document Title': node.metadata.get('file_name', 'Unknown'),
                'Page': node.metadata.get('page_label', 'N/A'),
                #'Formatted Text': formatted_text
            })

        # Display the formatted response
        #formatted_response = f"{response_data['Response Summary']}\n\n"
        #formatted_response = "Source Documents:\n"

        formatted_response = ""
        for doc in response_data['Source Documents']:
            # Build HTML formatted content for each document
            formatted_response += f"<li><strong>Document:</strong> {doc['Document Title']} <strong>(Page: {doc['Page']})</strong></li>\n"
            #formatted_response += f"Content:\n{doc['Formatted Text']}\n\n"

        response_txt = str(response)
        response_txt += f"<h3>Source Documents</h3>\n<ul>\n{formatted_response}</ul>"
        print(formatted_response)

        #response += formatted_response
        # Convert and print HTML
        html_response = markdown_to_html(response_txt)
        print(html_response)
        state.response = html_response
        logger.debug(f"Generated response for query: {state.query}")
        yield

    except IndexError as e:
        state.response = "Data is still loading, please try again shortly."
        logger.error(f"Index error: {e}")


def markdown_to_html(response):
    # Convert markdown text to HTML
    html_output = markdown.markdown(response)
    return html_output

def display_response():
    state = me.state(State)
    if state.response is not None:
        print(state.response)

        with me.box(
            style=me.Style(
                padding=me.Padding.all(16),  
                background="#F8F3FA",        
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
                me.html(state.response,
                    style=me.Style(
                    font_size=16,
                    line_height="1.5",
                    ),
                )


