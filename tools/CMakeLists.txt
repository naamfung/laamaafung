# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    if (GGML_METAL)
        add_subdirectory(tuning)
    endif()
    add_subdirectory(results)

    if (LLAMA_KVMEM)
        add_executable(llama-kvmem-cli
            llama-kvmem-cli.cpp
            kvmem-spec.cpp
        )
        target_include_directories(llama-kvmem-cli PRIVATE
            ${LLAMA_KVMEM_ROOT}/src/adapter
            ${LLAMA_KVMEM_ROOT}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/../src
        )
        target_link_libraries(llama-kvmem-cli PRIVATE llama llama-common)
        target_compile_features(llama-kvmem-cli PRIVATE cxx_std_17)

        add_executable(llama-kvmem-server
            llama-kvmem-server.cpp
            kvmem-spec.cpp
            kvmem-vision.cpp
            server/server-common.cpp
        )
        target_include_directories(llama-kvmem-server PRIVATE
            ${LLAMA_KVMEM_ROOT}/src/adapter
            ${LLAMA_KVMEM_ROOT}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/../src
            ${CMAKE_CURRENT_SOURCE_DIR}/server
            ${CMAKE_CURRENT_SOURCE_DIR}/mtmd
            ${CMAKE_CURRENT_SOURCE_DIR}/../common
            ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
        )
        target_link_libraries(llama-kvmem-server PRIVATE llama llama-common cpp-httplib mtmd)
        target_compile_features(llama-kvmem-server PRIVATE cxx_std_17)
        # This target compiles its own copy of server/server-common.cpp, which guards its
        # KVMem hooks with `#if defined(LLAMA_KVMEM)`. Without the define those hooks are
        # silently compiled out here (llama_kvmem_image_token_cap, i.e. the pre-tokenization
        # image downscale) even though the tool itself is KVMem-only and calls the adapter
        # unconditionally, so the define has to be set per target.
        target_compile_definitions(llama-kvmem-server PRIVATE LLAMA_KVMEM=1)
    endif()
endif()
