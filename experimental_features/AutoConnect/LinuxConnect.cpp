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

struct CameraInfo {
    crl::multisense::system::DeviceInfo devInfo;
    crl::multisense::image::Config imgConf;
    crl::multisense::system::NetworkConfig netConfig;
    crl::multisense::system::VersionInfo versionInfo;
    crl::multisense::image::Calibration camCal{};
    std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes;
    crl::multisense::DataSource supportedSources{0};
    std::vector<uint8_t *> rawImages;
    int sensorMTU = 0;
} cameraInfo;

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
            std::cerr << "Skipping Adapter: " << i->if_name << "| Reason: ioctl fail | " << strerror(errno)
                      << std::endl;
            continue;

        }

        if (ecmd.req.link_mode_masks_nwords >= 0 || ecmd.req.cmd != ETHTOOL_GLINKSETTINGS) {
            continue;
        }

        ecmd.req.link_mode_masks_nwords = -ecmd.req.link_mode_masks_nwords;

        if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
            std::cerr << "ioctl fail: " << strerror(errno) << std::endl;
            continue;
        }

        std::cout << "\n\n\tFound Adapter: " << i->if_name
                  << "\n\tSpeed: " << ecmd.req.speed
                  << "\n\tDuplex: " << static_cast<int>(ecmd.req.duplex)
                  << "\n\tPort: " << static_cast<int>(ecmd.req.port)
                  << std::endl;

        adapterSupportResult.back().supports = true;

    }


    close(fd);
    if_freenameindex(ifn);
    return adapterSupportResult;
}


/*
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
*/

int icmp = 0, others = 0, igmp = 0, total = 0;

void ProcessPacket(unsigned char *buffer, int size, std::string *address) {
//Get the IP Header part of this packet , excluding the ethernet header
    auto *iph = (struct iphdr *) (buffer + sizeof(struct ethhdr));
    struct in_addr ip_addr{};

    ++total;
    switch (iph->protocol) //Check the Protocol and do accordingly...
    {
        case 1: //ICMP Protocol
            ++icmp;
            break;
        case 2: //IGMP Protocol
            ++igmp;
            ip_addr.s_addr = iph->saddr;
            *address = inet_ntoa(ip_addr);
            printf("ICMP : %d IGMP : %d Others : %d Total : %d\n", icmp, igmp, others, total);

            break;
        case 6: //TCP
            break;
        case 17: // UDP
            break;
        default: //Some Other Protocol like ARP etc.
            ++others;
            break;
    }

}

int main(int argc, char *argv[]) {
    // Get list of network adapters that are  supports our application
    std::vector<AdapterSupportResult> adapterSupportResult = checkNetworkAdapterSupport();
    std::string hostAddress;
    int i = 0;

    // Loop keeps retrying to connect on supported network adapters.
    while (true) {
        auto adapter = adapterSupportResult[i];
        i++;
        if (i == adapterSupportResult.size())
            i = 0;

        if (!adapter.supports) {
            continue;
        }
        printf("\nTesting Adapter: %s\n", adapter.name.c_str());
        int sd = -1;
        // Submit request for a socket descriptor to look up interface.
        if ((sd = socket(PF_INET, SOCK_RAW, IPPROTO_RAW)) < 0) {
            perror("socket() failed to get socket descriptor for using ioctl() ");
            exit(EXIT_FAILURE);
        }

        /* set the network card in promiscuos mode*/
        // An ioctl() request has encoded in it whether the argument is an in parameter or out parameter
        // SIOCGIFFLAGS	0x8913		/* get flags			*/
        // SIOCSIFFLAGS	0x8914		/* set flags			*/
        struct ifreq ethreq;
        strncpy(ethreq.ifr_name, adapter.name.c_str(), IF_NAMESIZE);
        if (ioctl(sd, SIOCGIFFLAGS, &ethreq) == -1) {
            perror("ioctl");
            close(sd);
            exit(1);
        }
        ethreq.ifr_flags |= IFF_PROMISC;
        if (ioctl(sd, SIOCSIFFLAGS, &ethreq) == -1) {
            perror("ioctl");
            close(sd);
            exit(1);
        }

        int saddr_size, data_size;
        struct sockaddr saddr{};
        auto *buffer = (unsigned char *) malloc(IP_MAXPACKET + 1);

        int sock_raw = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
        if (sock_raw < 0) {
            //Print the error with proper message
            perror("Socket Error");
            return 1;
        }
        int ret = setsockopt(sock_raw, SOL_SOCKET, SO_BINDTODEVICE, adapter.name.c_str(), adapter.name.length() + 1);
        if (ret != 0) {
            std::cerr << "Failed to bind to network adapter" << std::endl;
            continue;
        }


        std::string ipAddress;
        printf("Listening for a IGMP packet\n");
        while (1) {
            saddr_size = sizeof saddr;
            //Receive a packet
            data_size = (int) recvfrom(sock_raw, buffer, IP_MAXPACKET, 0, &saddr,
                                       (socklen_t *) &saddr_size);

            if (data_size < 0) {
                printf("Recvfrom error , failed to get packets\n");
                return 1;
            }
            //Now process the packet
            ProcessPacket(buffer, data_size, &ipAddress);

            if (!ipAddress.empty()) {
                close(sock_raw);
                break;
            }
        }
        printf("Found address: %s\n", ipAddress.c_str());

        // Set the host ip address to the same subnet but with *.1 at the end.
        hostAddress = ipAddress;
        std::string last_element(hostAddress.substr(hostAddress.rfind(".")));
        auto ptr = hostAddress.rfind(".");
        hostAddress.replace(ptr, last_element.length(), ".1");
        printf("Setting host address to: %s\n", hostAddress.c_str());



        /*** CALL IOCTL Operations to set the address of the adapter/socket  ***/
        // Create the socket.
        int camera_fd = -1;
        camera_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (camera_fd < 0)
            fprintf(stderr, "failed to create the UDP socket: %s",
                    strerror(errno));

        // Bind Camera FD to the ethernet device
        const char *interface = adapter.name.c_str();
        ret = setsockopt(camera_fd, SOL_SOCKET, SO_BINDTODEVICE, interface,
                         15); // 15 is max length for an adapter name.
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
        memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
        int ioctl_result = ioctl(camera_fd, SIOCSIFADDR, &ifr);  // Set IP address
        if (ioctl_result < 0) {
            fprintf(stderr, "ioctl SIOCSIFADDR: %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }

        /// put mask in ifr structure
        memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
        ioctl_result = ioctl(camera_fd, SIOCSIFNETMASK, &ifr);   // Set subnet mask
        if (ioctl_result < 0) {
            fprintf(stderr, "ioctl SIOCSIFNETMASK: ");
            perror("");
            exit(EXIT_FAILURE);
        }
        /*** END **/

        // Attempt to connect to camera and post some info
        auto *cameraInterface = crl::multisense::Channel::Create(ipAddress);

        if (cameraInterface != nullptr) {
            cameraInterface->getImageConfig(cameraInfo.imgConf);
            cameraInterface->getNetworkConfig(cameraInfo.netConfig);
            cameraInterface->getVersionInfo(cameraInfo.versionInfo);
            cameraInterface->getDeviceInfo(cameraInfo.devInfo);
            cameraInterface->getDeviceModes(cameraInfo.supportedDeviceModes);
            cameraInterface->getImageCalibration(cameraInfo.camCal);
            cameraInterface->getEnabledStreams(cameraInfo.supportedSources);
            cameraInterface->getMtu(cameraInfo.sensorMTU);
            break;
        } else {
            printf("Did not find a camera on %s\n Retrying...\n", ipAddress.c_str());
            close(camera_fd);
        }

    }

    printf("\nConnected to camera: %s\n Camera IP (From LibMultiSense): %s\n Camera netmask %s\n Host IP: %s\n Camera FW build date: %s\n Build date %u\n",
           cameraInfo.devInfo.name.c_str(), cameraInfo.netConfig.ipv4Address.c_str(),
           cameraInfo.netConfig.ipv4Netmask.c_str(), hostAddress.c_str(),
           cameraInfo.versionInfo.sensorFirmwareBuildDate.c_str(),
           cameraInfo.versionInfo.sensorFirmwareVersion);


    exit(EXIT_SUCCESS);
}
