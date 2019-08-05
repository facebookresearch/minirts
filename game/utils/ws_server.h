// File: ws_server.h

#pragma once
#include <iostream>
#include <memory>
#include <thread>

#define ASIO_STANDALONE
#include "ws_config.h"
#include "websocketpp/server.hpp"

namespace {

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;
typedef websocketpp::server<websocketpp::config::elf_asio> Server;

// pull out the type of messages sent by our config
typedef Server::message_ptr message_ptr;
}

class WSServerException : public std::exception {
 public:
  WSServerException(int port) : port_(port) {}

  const char* what() const throw () {
    return "client disconnected";
  }

  int get_port() const { return port_; }

 private:
  int port_;
};

// A simple websocket server with only one client
class WSServer {
 public:
  ~WSServer() {
    server_.close(
        *hdl_.get(), websocketpp::close::status::normal, "game's over");
    server_.stop_listening();
    th_.join();
  }

  WSServer(
      int port,
      std::function<void(const std::string&)> callback,
      bool connect_on_start = true)
      : port_(port), callback_{callback} {
    try {
      // Set logging settings -- no log
      server_.clear_access_channels(websocketpp::log::alevel::all);

      server_.set_reuse_addr(true);

      // Initialize Asio
      server_.init_asio();

      // Register our message handler
      server_.set_message_handler(
          bind(&WSServer::on_message, this, ::_1, ::_2));
      server_.set_open_handler(bind(&WSServer::on_open, this, ::_1));

      server_.listen(port);

      // Start the server accept loop
      server_.start_accept();

      // Start the ASIO io_service run loop
      th_ = std::thread([&]() { server_.run(); });

      if (connect_on_start) {
        // wait for the connection
        std::cout << "Waiting for websocket client ..." << std::endl;
        while (true) {
          if (hdl_)
            break;
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        auto conn = server_.get_con_from_hdl(*hdl_.get());
        std::cout << "Connected to client " << conn->get_host() << "."
                  << std::endl;
      }
    } catch (websocketpp::exception const& e) {
      throw WSServerException(port_);
    } catch (...) {
      std::cerr << "Websocket Server receive exception: " << std::endl;
      abort();
    }
  };

  bool ready() const {
    return bool(hdl_);
  }

  void on_open(websocketpp::connection_hdl hdl) {
    assert(!hdl_); // we only allow single client
    hdl_.reset(new websocketpp::connection_hdl{hdl});
  }

  // Define a callback to handle incoming messages
  void on_message(websocketpp::connection_hdl, message_ptr msg) {
    try {
      std::string payload = msg->get_payload();
      callback_(payload);
    } catch (websocketpp::exception const& e) {
      throw WSServerException(port_);
    }
  }

  void send(const std::string& msg) {
    try {
      server_.send(*(hdl_.get()), msg, websocketpp::frame::opcode::TEXT);
    } catch (websocketpp::exception const& e) {
      throw WSServerException(port_);
    }
  }

 protected:
  Server server_;
  std::unique_ptr<websocketpp::connection_hdl> hdl_;
  std::thread th_;
  int port_;
  std::function<void(const std::string&)> callback_;
};
