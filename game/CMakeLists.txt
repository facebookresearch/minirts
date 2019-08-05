# auto-gen some files
set(CMD_COMPILE_SCRIPT
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_cmds.py"
  CACHE STRING "Path to the python script to compile commands.")

function(add_rts_command_gen DEF NAME)
  # use the specified python, if any
  if (NOT PYTHON_EXECUTABLE)
    set(_PYTHON_EXE python)
  else()
    set(_PYTHON_EXE "${PYTHON_EXECUTABLE}")
  endif()

  add_custom_command(
    OUTPUT ${DEF}.gen.h
    COMMAND ${_PYTHON_EXE} ${CMD_COMPILE_SCRIPT} --def_file ${DEF} --name ${NAME}
    COMMENT "Running compile_cmds.py on ${DEF}"
    VERBATIM
    DEPENDS ${DEF}.def
    )
  set_source_files_properties(${DEF}.gen.h PROPERTIES GENERATED TRUE)
endfunction()

add_rts_command_gen(${CMAKE_CURRENT_SOURCE_DIR}/engine/cmd engine)
add_rts_command_gen(${CMAKE_CURRENT_SOURCE_DIR}/engine/cmd_specific engine_specific)
add_rts_command_gen(${CMAKE_CURRENT_SOURCE_DIR}/game_MC/cmd_specific minirts_specific)

# files for lib
file(GLOB_RECURSE ENGINE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/engine/*.cc)
file(GLOB_RECURSE MC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/game_MC/*.cc)
set(AI_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/ai_factory.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/comm_ai.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/raw2cmd.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/replay_loader.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/rule_ai.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/save2json.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/rule_extractor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/cmd_reply.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/state_dumper.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ai/executor_extractor.cc
)

# find necessary package
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# add lib and link
add_library(_minirts ${ENGINE_SOURCES} ${MC_SOURCES} ${AI_SOURCES})
target_include_directories(_minirts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  _minirts
  PUBLIC
  concurrentqueue
  websocketpp
  selene
  json
  _tube
)

# dependency to make sure files will be generated
add_custom_target(gen-headers DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/cmd.gen.h
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/cmd_specific.gen.h
        ${CMAKE_CURRENT_SOURCE_DIR}/game_MC/cmd_specific.gen.h)
add_dependencies(_minirts gen-headers)

# pybind lib
pybind11_add_module(minirts pybind.cc)
target_link_libraries(
  minirts
  PUBLIC
  _minirts
)

# add executable
find_package(Threads)
add_executable(minirts-backend ${CMAKE_CURRENT_SOURCE_DIR}/backend/main_loop.cc)
target_link_libraries(
  minirts-backend
  PUBLIC
  _minirts
  ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(
  minirts-backend
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}
)


# link lua for runtime
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}/game_MC/lua/map_generator.lua
  ${CMAKE_BINARY_DIR}/map_generator.lua)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}/game_MC/lua/unit_generator.lua
  ${CMAKE_BINARY_DIR}/unit_generator.lua)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}/game_MC/lua/unit_factory.lua
  ${CMAKE_BINARY_DIR}/unit_factory.lua)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}/game_MC/lua/attack_rule_book.lua
  ${CMAKE_BINARY_DIR}/attack_rule_book.lua)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}/game_MC/lua/random.lua
  ${CMAKE_BINARY_DIR}/random.lua)
