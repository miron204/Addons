FROM ghcr.io/home-assistant/aarch64-base:latest

RUN apk add --no-cache python3 py3-pip

WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt --break-system-packages

COPY rootfs /
COPY deepgram_server.py /app

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10301 \
        | grep -q "deepgram_server" \
        || exit 1
