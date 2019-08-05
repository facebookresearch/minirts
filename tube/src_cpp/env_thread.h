#pragma once

namespace tube {

class EnvThread {
 public:
  virtual ~EnvThread() {
  }

  virtual void mainLoop() = 0;
};
}
