// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once
#include "concurrentqueue.h"

#include "utils/ws_server.h"

#include "ai/ai.h"
#include "ai/replay_loader.h"
#include "ai/raw2cmd.h"
#include "ai/save2json.h"

class WebCtrl {
 public:
  WebCtrl(int port, bool connect_on_start = true) {
    server_.reset(new WSServer{
        port,
        [this](const std::string& msg) { this->queue_.enqueue(msg); },
        connect_on_start});
  }

  PlayerId GetId() const {
    return _raw_converter.GetPlayerId();
  }

  void setID(PlayerId id) {
    _raw_converter.setID(id);
  }

  void Receive(
      const RTSStateExtend& s,
      std::vector<CmdBPtr>* cmds,
      std::vector<UICmd>* ui_cmds);

  void Send(const std::string& s) {
    server_->send(s);
  }

  void Extract(const RTSStateExtend& s, json* game);

  void ExtractWithId(const RTSStateExtend& s, int player_id, json* game);

  bool Ready() const;

 private:
  RawToCmd _raw_converter;
  std::unique_ptr<WSServer> server_;
  moodycamel::ConcurrentQueue<std::string> queue_;
};

class TCPAI : public AI {
 public:
  TCPAI(int port)
      : AI(AIOption(), 0), port(port), _ctrl(port, false)
  {}

  virtual void setId(int id) override {
    AI::setId(id);
    _ctrl.setID(getId());
  }

  virtual bool act(const RTSStateExtend& state, RTSAction* action) override;

  virtual bool isReady() const {
    return _ctrl.Ready();
  }

  const int port = -1;
 protected:
  WebCtrl _ctrl;

 // protected:
};

class TCPPlayerAI : public TCPAI {
 public:
  TCPPlayerAI(int port)
      : TCPAI(port)
  {}

  virtual bool act(const RTSStateExtend& state, RTSAction* action) override;
  // bool act(const State& s, RTSAction* action) override;

  // bool isReady() const override {
  //   return _ctrl.Ready();
  // }

 // private:
 //  WebCtrl _ctrl;

 // protected:
 //  void onSetID() override {
 //    this->AI::onSetID();
 //    _ctrl.setID(getId());
 //  }
};

class TCPCoachAI : public TCPAI {
 public:
  TCPCoachAI(int port, int player_id)
      : TCPAI(port), _player_id(player_id)
  {}

  virtual bool act(const RTSStateExtend& state, RTSAction* action) override;

  // bool act(const State& s, RTSAction* action) override;

  // bool isReady() const override {
  //   return _ctrl.Ready();
  // }

 private:
 //  WebCtrl _ctrl;
  int _player_id;

 // protected:
 //  void onSetID() override {
 //    this->AI::onSetID();
 //    _ctrl.setID(getId());
 //  }
};

class TCPSpectator : public Replayer {
 public:
  using Action = typename Replayer::Action;

  TCPSpectator(const std::string& replay_filename, int vis_after, int port)
      : Replayer(replay_filename),
        _ctrl(port),
        _vis_after(vis_after)
  {
    _ctrl.setID(INVALID);
  }

  bool act(const RTSStateExtend& s, Action* action) override;

 private:
  bool extractSelectedUnits(json* game);
  bool extractMouseActions(json* game);

  WebCtrl _ctrl;
  int _vis_after;

  std::vector<std::string> _history_states;
};
