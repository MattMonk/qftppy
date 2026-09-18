cmake_minimum_required(VERSION 3.24)
project(qftcpp_ref LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(QFTPP_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/patches/tensorindex-permute.patch)

include(FetchContent)
FetchContent_Declare(
  qftpp
  GIT_REPOSITORY https://github.com/jdalseno/qft.git
  GIT_TAG df048f3a8c33ffb8984671fb057acd7dcd345a61
  # Upstream TensorIndex::Permute never terminates for rank 4, which hangs Tensor::Symmetric()
  # (orbital tensors of rank >= 4, spin-2 projectors). See patches/make_patch.py.
  # The check makes this idempotent: re-running a build (e.g. after `pixi reinstall`) re-applies
  # the patch only if it is not already applied.
  PATCH_COMMAND sh -c "git apply --reverse --check ${QFTPP_PATCH} || git apply ${QFTPP_PATCH}"
  UPDATE_DISCONNECTED ON
)
FetchContent_MakeAvailable(qftpp)

file(GLOB QFTPP_SOURCES
  ${qftpp_SOURCE_DIR}/src/tensor/*.C
  ${qftpp_SOURCE_DIR}/src/relativistic-quantum-mechanics/*.C)
set_source_files_properties(${QFTPP_SOURCES} PROPERTIES LANGUAGE CXX)
add_library(qftpp STATIC ${QFTPP_SOURCES})
target_include_directories(qftpp PUBLIC ${qftpp_SOURCE_DIR}/include)
target_compile_options(qftpp PRIVATE -w)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(qftcpp_ref bindings.cpp)
target_link_libraries(qftcpp_ref PRIVATE qftpp)
install(TARGETS qftcpp_ref DESTINATION .)
