FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential cmake git libzstd-dev libsdl2-dev libgl1-mesa-glx libgl1-mesa-dev libvulkan1 libvulkan-dev libassimp-dev opencl-c-headers libfmt-dev pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libtbb-dev libssl-dev && \
    apt-get install -y wget && \
    wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.216-focal.list https://packages.lunarg.com/vulkan/1.3.216/lunarg-vulkan-1.3.216-focal.list && \
    apt-get update && \
    apt-get install -y vulkan-sdk && \
    git clone --recurse-submodule https://github.com/carnegierobotics/multisense_viewer

CMD ["/bin/bash"]