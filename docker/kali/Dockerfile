FROM kalilinux/kali-rolling

RUN apt-get update && \
    apt-get install -y hping3 curl iputils-ping && \
    apt-get clean

CMD ["/bin/bash"]