#include <netdb.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <armadillo>
#include <arpa/inet.h>
#include <fitsio.h>

#define SERVER_PORT "7777"

int main(int argc, char **argv) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock == -1) {
        std::cout << "socket error: " << errno << std::endl;
        exit(-1);
    }
    std::cout << "socket created!" << std::endl;

    int so_broadcast;
    so_broadcast = TRUE;
    int ret = -1;
    if ((ret = setsockopt(sock,
                          SOL_SOCKET,
                          SO_BROADCAST,
                          &so_broadcast,
                          sizeof so_broadcast)) != 0) {

        std::cout << "setsockopt error: " << std::endl;
        exit(-1);
    }
    struct sockaddr_in recvAddr;
    struct sockaddr_in Sender_addr;

    int len = sizeof(struct sockaddr_in);
    char recvbuff[50];
    int recvbufflen = 50;
    char sendMSG[] = "Broadcast message from READER";

    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(9009);
    recvAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (sockaddr *) &recvAddr, sizeof(recvAddr)) < 0) {
        std::cout << "Error in BINDING" << std::endl;
        return 0;
    }
    std::cout << "binded successfully!" << std::endl;

    recvfrom(sock, recvbuff, recvbufflen, 0, (sockaddr *) &Sender_addr, reinterpret_cast<socklen_t *>(&len));


    std::cout << "\n\n\tReceived Message is : " << recvbuff;
    close(sock);
}