FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Build args make container-written files (sweep.csv, field.bin, *.html, build/)
# owned by your host user instead of root, so they're not root-locked on your laptop.
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=worker



ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libgomp1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the Plotly renderer.
RUN pip3 install --no-cache-dir numpy plotly

# Create a non-root user matching the host UID/GID.
RUN groupadd -g ${GROUP_ID} ${USER_NAME} || groupadd ${USER_NAME} && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m ${USER_NAME}

WORKDIR /app
RUN chown ${USER_NAME}:${USER_NAME} /app
USER ${USER_NAME}