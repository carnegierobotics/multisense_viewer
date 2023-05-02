//
// Created by magnus on 4/10/23.
//

#ifndef MULTISENSE_VIEWER_SERVERCONNECTION_H
#define MULTISENSE_VIEWER_SERVERCONNECTION_H

#include <httplib.h>
#include "Viewer/Core/RendererConfig.h"

namespace VkRender {
class ServerConnection {
public:
/**
 *
 * @param identifier anonymous string to identify this specific installation of the viewer application
 * @param ip and port for server
 * @param protocol http or https
 * @param destination destination in the server that requests should go to by default
 */
    ServerConnection(std::string identifier, VkRender::RendererConfig::CRLServerInfo serverInfo){
        // Create the HTTP client
        m_IP = serverInfo.server;
        m_Protocol = serverInfo.protocol;
        m_Destination = serverInfo.destination;
        m_Identifier = identifier;

        m_Client = std::make_unique<httplib::Client>(serverInfo.protocol + "://" + serverInfo.server);

        m_Client->set_connection_timeout(3, 0); // 3 seconds
        m_Client->set_read_timeout(5, 0); // 5 seconds
        m_Client->set_write_timeout(5, 0); // 5 seconds
    }

    ~ServerConnection(){

    }

    void sendUsageStatistics(const std::filesystem::path& usageFilePath, std::filesystem::path logFilePath);

private:
    std::unique_ptr<httplib::Client> m_Client;
    std::string m_Identifier;

    std::string m_Protocol;
    std::string m_IP;
    std::string m_Destination;
};
}

#endif //MULTISENSE_VIEWER_SERVERCONNECTION_H
