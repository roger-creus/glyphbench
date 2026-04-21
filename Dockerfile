# ATLAS RL — minimal vLLM-based container for cluster eval
#
# Build:  docker build -t glyphbench:latest .
# To SIF: apptainer build glyphbench.sif docker-daemon://glyphbench:latest

FROM vllm/vllm-openai:latest

# Install glyphbench dependencies (gymnasium, pydantic, etc.)
RUN pip install --no-cache-dir \
    gymnasium>=1.0 \
    numpy>=2.0 \
    pydantic>=2.9 \
    pyyaml>=6.0 \
    jinja2>=3.1 \
    rich>=13.9

# Project code will be bind-mounted at /src via apptainer
# PYTHONPATH=/src/src set at runtime
WORKDIR /src
