# add_subdirectory(json)
add_library(json INTERFACE)
target_include_directories(json SYSTEM INTERFACE
  json/single_include
)

# asio library
add_library(asio INTERFACE)
target_include_directories(asio SYSTEM INTERFACE
  asio
)

# websocketapp library
add_library(websocketpp INTERFACE)
target_include_directories(websocketpp SYSTEM INTERFACE
  websocketpp
)
target_link_libraries(websocketpp INTERFACE asio)

find_package(Lua REQUIRED)

# selene library
add_library(selene INTERFACE)
target_include_directories(selene SYSTEM INTERFACE
  Selene
  ${LUA_INCLUDE_DIR}
)
target_link_libraries(selene INTERFACE ${LUA_LIBRARIES})
# set_target_properties(selene PROPERTIES COMPILE_FLAGS "")

add_library(concurrentqueue INTERFACE)
target_include_directories(concurrentqueue INTERFACE
  concurrentqueue
)
