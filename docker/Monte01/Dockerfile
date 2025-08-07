FROM ghcr.io/linkage-x/hirol-robots:v0

RUN apt update && apt install -y libiceoryx-binding-c1 libiceoryx-binding-c-dev --fix-missing
RUN export RMW_IMPLEMENTATION=sudo apt update