FROM ubuntu:latest
LABEL authors="delar"

ENTRYPOINT ["top", "-b"]