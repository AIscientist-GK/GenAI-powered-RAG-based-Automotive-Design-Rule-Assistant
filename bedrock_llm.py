from langchain_community.chat_models import BedrockChat
import boto3

# === Get Claude Haiku from Bedrock ===
def get_bedrock_llm():
    bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-south-1", verify=False)
    return BedrockChat(
        model_id="mistral.mixtral-8x7b-instruct-v0:1",
        client=bedrock_runtime,
        model_kwargs={ "temperature": 0.7}
    )





