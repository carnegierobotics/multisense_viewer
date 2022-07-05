#include <arpa/inet.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <cstdlib>
#include <unistd.h>
#include <linux/if_link.h>
#include <cstdio>
#include <unistd.h>           // close()
#include <cstring>           // strcpy, memset()

#include <netinet/ip.h>       // IP_MAXPACKET (65535)
#include <sys/types.h>        // needed for socket(), uint8_t, uint16_t
#include <sys/socket.h>       // needed for socket()
#include <linux/if_ether.h>   // ETH_P_ARP = 0x0806, ETH_P_ALL = 0x0003
#include <net/ethernet.h>

#include <cerrno>            // errno, perror()
#include <MultiSense/MultiSenseChannel.hh>
#include <net/if.h>
#include <sys/ioctl.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <iostream>

struct AdapterSupportResult {
    std::string name; // Name of network adapter tested
    bool supports; // 0: for bad, 1: for good

    AdapterSupportResult(const char *name, uint8_t supports) {
        this->name = name;
        this->supports = supports;
    }
};


std::vector<AdapterSupportResult>
checkNetworkAdapterSupport() {
    std::vector<AdapterSupportResult> adapterSupportResult;
    auto ifn = if_nameindex();
    auto fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    for (auto i = ifn; i->if_name; ++i) {
        struct {
            struct ethtool_link_settings req;
            __u32 link_mode_data[3 * 127];
        } ecmd{};

        adapterSupportResult.emplace_back(i->if_name, false);

        // Skip the loopback
        //if (i->if_index == 1) {
            //continue;
        //}

        auto ifr = ifreq{};
        std::strncpy(ifr.ifr_name, i->if_name, IF_NAMESIZE);

        ecmd.req.cmd = ETHTOOL_GLINKSETTINGS;
        ifr.ifr_data = reinterpret_cast<char *>(&ecmd);

        if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
            std::cerr << "ioctl fail: " << strerror(errno) << std::endl;
            adapterSupportResult.back().supports = false;
            continue;

        }

        if (ecmd.req.link_mode_masks_nwords >= 0 || ecmd.req.cmd != ETHTOOL_GLINKSETTINGS) {
            adapterSupportResult.back().supports = false;
            continue;
        }

        ecmd.req.link_mode_masks_nwords = -ecmd.req.link_mode_masks_nwords;

        if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
            std::cerr << "ioctl fail: " << strerror(errno) << std::endl;
            adapterSupportResult.back().supports = false;
            continue;
        }

        std::cout << "\tSpeed: " << ecmd.req.speed
                  << "\n\tDuplex: " << static_cast<int>(ecmd.req.duplex)
                  << "\n\tPort: " << static_cast<int>(ecmd.req.port)
                  << std::endl;

        adapterSupportResult.back().supports = true;

    }



    close(fd);
    if_freenameindex(ifn);
    return adapterSupportResult;
}




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

std::string listenForARP(const char* interface) {
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

    int ret = setsockopt( sd, SOL_SOCKET, SO_BINDTODEVICE, interface, 16);
    if (ret != 0){
        printf("Error binding to adapter\n");
        exit(EXIT_FAILURE);

    }
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
    std::string address = std::to_string(arphdr->sender_ip[0]) + "." + std::to_string(arphdr->sender_ip[1]) + "." + std::to_string(arphdr->sender_ip[2]) + "." +std::to_string(arphdr->sender_ip[3]);
    printf("Target (this node) hardware (MAC) address: ");
    for (i = 0; i < 5; i++) {
        printf("%02x:", arphdr->target_mac[i]);
    }
    printf("%02x\n", arphdr->target_mac[5]);
    printf("Target (this node) protocol (IPv4) address: %u.%u.%u.%u\n",
           arphdr->target_ip[0], arphdr->target_ip[1], arphdr->target_ip[2], arphdr->target_ip[3]);

    free(ether_frame);

    return address;
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


int main(int argc, char *argv[]) {
    struct ifaddrs *ifaddr;
    int family, s;
    char host[NI_MAXHOST];

    // Get list of network adapters that are ethernet i.e. supports our application
    std::vector<AdapterSupportResult> adapterSupportResult = checkNetworkAdapterSupport();

    //Ping the list of adapters for IP ARP requests.
    int camera_fd = 0;
    //
    // Create the socket.
    camera_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (camera_fd < 0)
        fprintf(stderr, "failed to create the UDP socket: %s",
                strerror(errno));
    int i = 0;
    while (true){
        if (i == adapterSupportResult.size())
            i = 0;
        auto adapter = adapterSupportResult[i];
        i++;

        if (!adapter.supports){
            continue;
        }
        printf("Testing Adapter: %s\n", adapter.name.c_str());

        std::string ipAddress = listenForARP(adapter.name.c_str());
        printf("Found address: %s\n", ipAddress.c_str());

        std::string hostAddress = ipAddress;
        int lastTwoDigits = 0;
        int n = 2;
        try {
            lastTwoDigits = std::stoi(ipAddress.substr(ipAddress.length() - 2));
        }
        catch(...) {
            printf("Ip format error\n");
            n = 1;
            lastTwoDigits = std::stoi(ipAddress.substr(ipAddress.length() - n));
        }
        lastTwoDigits++;
        hostAddress.replace(hostAddress.size() - n, n, std::to_string(lastTwoDigits));

        printf("Setting host address to: %s\n", hostAddress.c_str());


        // Specify interface name
        const char *interface = adapter.name.c_str();
        int ret = setsockopt(camera_fd, SOL_SOCKET, SO_BINDTODEVICE, interface, 15); // 15 is max length for an adapter name.

        if (ret != 0) {
            fprintf(stderr, "Error binding to: %s, %s", interface, strerror(errno));
            continue;
        }

        struct ifreq ifr{};
        /// note: no pointer here
        struct sockaddr_in inet_addr{}, subnet_mask{};
        /* get interface name */
        /* Prepare the struct ifreq */
        bzero(ifr.ifr_name, IFNAMSIZ);
        strncpy(ifr.ifr_name, interface, IFNAMSIZ);

        /// note: prepare the two struct sockaddr_in
        inet_addr.sin_family = AF_INET;
        int inet_addr_config_result = inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

        subnet_mask.sin_family = AF_INET;
        int subnet_mask_config_result = inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));

        // Call ioctl to configure network devices
        /// put addr in ifr structure
        memcpy(&(ifr.ifr_addr), &inet_addr, sizeof (struct sockaddr));
        int ioctl_result = ioctl(camera_fd, SIOCSIFADDR, &ifr);  // Set IP address
        if(ioctl_result < 0)
        {
            fprintf(stderr, "ioctl SIOCSIFADDR: ");
            perror("");
            exit(EXIT_FAILURE);
        }

        /// put mask in ifr structure
        memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof (struct sockaddr));
        ioctl_result = ioctl(camera_fd, SIOCSIFNETMASK, &ifr);   // Set subnet mask
        if(ioctl_result < 0)
        {
            fprintf(stderr, "ioctl SIOCSIFNETMASK: ");
            perror("");
            exit(EXIT_FAILURE);
        }
        /*** END **/

        auto* cameraInterface = crl::multisense::Channel::Create(ipAddress);

        if (cameraInterface != nullptr)
            break;

    }


    exit(EXIT_SUCCESS);
}
