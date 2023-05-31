cmake_minimum_required(VERSION 3.15)
project(StarRocks)

set(CMAKE_CXX_STANDARD 17)

include_directories(
        be/
        be/src
        be/test

        // thrift/protobuf files
        gensrc/build
        gensrc/build/common
        gensrc/build/gen_cpp

        // header files of third-parties
        ../deps/gperftools
        ../deps/include
)
include_directories(SYSTEM "/var/local/thirdparty/installed/include")
include_directories(SYSTEM "/var/local/thirdparty/installed/gperftools/include")


add_executable(StarRocks
        be/src/service/starrocks_main.cpp)