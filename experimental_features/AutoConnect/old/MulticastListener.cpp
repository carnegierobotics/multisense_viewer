/**************************************************************/
/* Multicast listener (server)                                */
/*                                                            */
/* Activation using: {program name} {Multicast IP} {port}     */
/*   {program name} - This program name                       */
/*   {Multicast IP} - The IP address to listen to (Class D)   */
/*   {port} - The port hnumber to listen on                   */
/*                                                            */
/*                                                            */
/*                                                            */
/*                                                            */
/*                                                            */
/*                                                            */
/*                                                            */
/*                                                            */
/*                                                            */
/* This is free software released under the GPL license.      */
/* See the GNU GPL for details.                               */
/*                                                            */
/* (c) Juan-Mariano de Goyeneche. 1998, 1999.                 */
/**************************************************************/


#include <stdio.h>          /* printf(), snprintf() */
#include <stdlib.h>         /* strtol(), exit() */
#include <sys/types.h>
#include <sys/socket.h>     /* socket(), setsockopt(), bind(), recvfrom(), sendto() */
#include <errno.h>          /* perror() */
#include <netinet/in.h>     /* IPPROTO_IP, sockaddr_in, htons(), htonl() */
#include <arpa/inet.h>      /* inet_addr() */
#include <unistd.h>         /* fork(), sleep() */
#include <sys/utsname.h>    /* uname() */
#include <string.h>         /* memset() */

#define MAXLEN 1024


int main(int argc, char* argv[])
{
    u_char no = 0;
    u_int yes = 1;      /* Used with SO_REUSEADDR.
                             In Linux both u_int */
    /* and u_char are valid. */
    int send_s, recv_s;     /* Sockets for sending and receiving. */
    u_char ttl;
    struct sockaddr_in mcast_group;
    struct ip_mreq mreq;
    struct utsname name;
    int n;
    int len;
    struct sockaddr_in from;
    char message [MAXLEN+1];

    if (argc != 3) {
        fprintf(stderr, "Usage: %s mcast_group port\n", argv[0]);
        exit(1);
    }

    memset(&mcast_group, 0, sizeof(mcast_group));
    mcast_group.sin_family = AF_INET;
    mcast_group.sin_port = htons((unsigned short int)strtol(argv[2], NULL, 0));
    mcast_group.sin_addr.s_addr = inet_addr(argv[1]);

    if ( (recv_s=socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror ("recv socket");
        exit(1);
    }

    if (setsockopt(recv_s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) < 0) {
        perror("reuseaddr setsockopt");
        exit(1);
    }

    if (bind(recv_s, (struct sockaddr*)&mcast_group, sizeof(mcast_group)) < 0) {
        perror ("bind");
        exit(1);
    }

    /* Preparatios for using Multicast */
    mreq.imr_multiaddr = mcast_group.sin_addr;
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);

    /* Tell the kernel we want to join that multicast group. */
    if (setsockopt(recv_s, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        perror ("add_membership setsockopt");
        exit(1);
    }

    for (;;) {
        len=sizeof(from);
        if ((n=recvfrom(recv_s, message, MAXLEN, 0,
                        (struct sockaddr*)&from, reinterpret_cast<socklen_t *>(&len))) < 0) {
            perror ("recv");
            exit(1);
        }
        message[n] = '\0'; /* null-terminate string */
        printf("%s: Received message from %s, size=%d !!\n",
               name.nodename,
               inet_ntoa(from.sin_addr), n);
        printf("\t%s \n", message);
    }
}