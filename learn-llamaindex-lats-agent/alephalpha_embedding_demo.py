import os

from llama_index.embeddings.alephalpha import AlephAlphaEmbedding

os.environ["AA_TOKEN"] = os.getenv("APEPHA_ALPHA_EMBEDDING_API_KEY")


if __name__ == '__main__':

    # To customize your token, do this
    # otherwise it will lookup AA_TOKEN from your env variable
    # embed_model = AlephAlpha(token="")

    # with representation='query'

    # 需要api_key，不太好申请
    embed_model = AlephAlphaEmbedding(
        model="luminous-base",
        representation="Query",
    )

    embeddings = embed_model.get_text_embedding("Hello Aleph Alpha!")

    print(len(embeddings))
    print(embeddings[:5])