FROM ghcr.io/home-assistant/aarch64-base:latest

RUN apk add --no-cache python3 py3-pip
RUN pip3 install deepgram-sdk==3.* wyoming --break-system-packages

WORKDIR /app
COPY rootfs /

COPY deepgram_server.py /app

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10301 \
        | grep -q "deepgram_server" \
        || exit 1
