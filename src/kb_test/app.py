import streamlit as st
import glob
import boto3
import json
import pprint

kb_id = "OGRFZKIFIE"

## Initialize Bedrock
session = boto3.Session(
    profile_name='137484672202_EVLearningPaths',
    region_name='us-east-1'
)

print(session)

# bedrock = session.client('bedrock-runtime', region_name='us-east-1')
bedrock_agent = session.client('bedrock-agent', region_name='us-east-1')
bedrock_agent_runtime = session.client('bedrock-agent-runtime', region_name='us-east-1')
# knowledge_base = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)


# rag_config = {
#     "knowledgeBaseConfiguration": {
#         "knowledgeBaseId": kb_id,
#         "modelArn": "anthropic.claude-v2:1"
#     },
#     "type": "KNOWLEDGE_BASE"
# }
# response = bedrock_agent_runtime.retrieve_and_generate(
#     input={'text': 'What knowledge do we have about finance?'},
#     retrieveAndGenerateConfiguration=rag_config,
#     knowledgeBaseId=kb_id
# )


### MainPage

st.set_page_config(page_title="KB Test")
st.title("Knowledge-base Test")


input_text = st.chat_input("Query")

if input_text:
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={'text': input_text}
    )

    # st.text(pprint.pformat(knowledge_base))

    for item in response['retrievalResults']:
        st.text(pprint.pformat(item))
