FROM vllm/vllm-openai:v0.9.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Your required wheels
RUN pip3 install --no-cache-dir flash_attn==2.8.0.post2
RUN pip3 install --no-cache-dir transformers==4.51.3



