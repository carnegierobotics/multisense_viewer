/*  Copyright (C) 2011-2015  P.D. Buchan (pdbuchan@yahoo.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Receive an ARP reply and extract the MAC address
// of the sender of the ARP reply, as well as any other
// information stored in the ethernet frame.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>           // close()
#include <string.h>           // strcpy, memset()

#include <netinet/ip.h>       // IP_MAXPACKET (65535)
#include <sys/types.h>        // needed for socket(), uint8_t, uint16_t
#include <sys/socket.h>       // needed for socket()
#include <linux/if_ether.h>   // ETH_P_ARP = 0x0806, ETH_P_ALL = 0x0003
#include <net/ethernet.h>

#include <errno.h>            // errno, perror()

// Define an struct for ARP header
typedef struct arp_hdr arp_hdr;
struct arp_hdr {
    uint16_t htype;
    uint16_t ptype;
    uint8_t hlen;
    uint8_t plen;
    uint16_t opcode;
    uint8_t sender_mac[6];
    uint8_t sender_ip[4];
    uint8_t target_mac[6];
    uint8_t target_ip[4];
};

// Define some constants.
#define ARPOP_REPLY 1         // Taken from <linux/if_arp.h>

// Function prototypes
uint8_t *allocate_ustrmem(int);

int
main(int argc, char **argv) {
    int i, sd, status;
    uint8_t *ether_frame;
    arp_hdr *arphdr;

    // Allocate memory for various arrays.
    ether_frame = allocate_ustrmem(IP_MAXPACKET);

    // Submit request for a raw socket descriptor.
    if ((sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) < 0) {
        perror("socket() failed ");
        exit(EXIT_FAILURE);
    }

    const char* interface = "enx606d3cbfbd11";
    setsockopt( sd, SOL_SOCKET, SO_BINDTODEVICE, interface, 16);

    // Listen for incoming ethernet frame from socket sd.
    // We expect an ARP ethernet frame of the form:
    //     MAC (6 bytes) + MAC (6 bytes) + ethernet type (2 bytes)
    //     + ethernet data (ARP header) (28 bytes)
    // Keep at it until we get an ARP reply.
    arphdr = (arp_hdr *) (ether_frame + 6 + 6 + 2);
    while (((((ether_frame[12]) << 8) + ether_frame[13]) != ETH_P_ARP) || (ntohs(arphdr->opcode) != ARPOP_REPLY)) {
        if ((status = recv(sd, ether_frame, IP_MAXPACKET, 0)) < 0) {
            if (errno == EINTR) {
                memset(ether_frame, 0, IP_MAXPACKET * sizeof(uint8_t));
                continue;  // Something weird happened, but let's try again.
            } else {
                perror("recv() failed:");
                exit(EXIT_FAILURE);
            }
        }
    }
    close(sd);

    // Print out contents of received ethernet frame.
    printf("\nEthernet frame header:\n");
    printf("Destination MAC (this node): ");
    for (i = 0; i < 5; i++) {
        printf("%02x:", ether_frame[i]);
    }
    printf("%02x\n", ether_frame[5]);
    printf("Source MAC: ");
    for (i = 0; i < 5; i++) {
        printf("%02x:", ether_frame[i + 6]);
    }
    printf("%02x\n", ether_frame[11]);
    // Next is ethernet type code (ETH_P_ARP for ARP).
    // http://www.iana.org/assignments/ethernet-numbers
    printf("Ethernet type code (2054 = ARP): %u\n", ((ether_frame[12]) << 8) + ether_frame[13]);
    printf("\nEthernet data (ARP header):\n");
    printf("Hardware type (1 = ethernet (10 Mb)): %u\n", ntohs(arphdr->htype));
    printf("Protocol type (2048 for IPv4 addresses): %u\n", ntohs(arphdr->ptype));
    printf("Hardware (MAC) address length (bytes): %u\n", arphdr->hlen);
    printf("Protocol (IPv4) address length (bytes): %u\n", arphdr->plen);
    printf("Opcode (2 = ARP reply): %u\n", ntohs(arphdr->opcode));
    printf("Sender hardware (MAC) address: ");
    for (i = 0; i < 5; i++) {
        printf("%02x:", arphdr->sender_mac[i]);
    }
    printf("%02x\n", arphdr->sender_mac[5]);
    printf("Sender protocol (IPv4) address: %u.%u.%u.%u\n",
           arphdr->sender_ip[0], arphdr->sender_ip[1], arphdr->sender_ip[2], arphdr->sender_ip[3]);
    printf("Target (this node) hardware (MAC) address: ");
    for (i = 0; i < 5; i++) {
        printf("%02x:", arphdr->target_mac[i]);
    }
    printf("%02x\n", arphdr->target_mac[5]);
    printf("Target (this node) protocol (IPv4) address: %u.%u.%u.%u\n",
           arphdr->target_ip[0], arphdr->target_ip[1], arphdr->target_ip[2], arphdr->target_ip[3]);

    free(ether_frame);

    return (EXIT_SUCCESS);
}

// Allocate memory for an array of unsigned chars.
uint8_t *
allocate_ustrmem(int len) {
    void *tmp;

    if (len <= 0) {
        fprintf(stderr, "ERROR: Cannot allocate memory because len = %i in allocate_ustrmem().\n", len);
        exit(EXIT_FAILURE);
    }

    tmp = (uint8_t *) malloc(len * sizeof(uint8_t));
    if (tmp != NULL) {
        memset(tmp, 0, len * sizeof(uint8_t));
        return static_cast<uint8_t *>(tmp);
    } else {
        fprintf(stderr, "ERROR: Cannot allocate memory for array allocate_ustrmem().\n");
        exit(EXIT_FAILURE);
    }
}