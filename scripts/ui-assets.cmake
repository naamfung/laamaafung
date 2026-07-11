# Provision UI assets and generate ui.cpp/ui.h.
#
# Asset provisioning priority:
#   1. Pre-built assets in SRC_DIST_DIR (manually built by user)
#   2. If BUILD_UI=ON: bun build (falls back to npm if bun not found)
#   3. If above did not produce assets: extract from local archive in
#      ${LLAMA_SOURCE_DIR}/files (llama-b<version>-ui.tar.gz)

cmake_minimum_required(VERSION 3.18)

set(UI_SOURCE_DIR     "" CACHE STRING "UI source directory (to run npm build)")
set(UI_BINARY_DIR     "" CACHE STRING "UI binary directory (to store generated files)")
set(LLAMA_SOURCE_DIR  "" CACHE STRING "Project source root (to resolve version from git)")
set(HF_BUCKET         "" CACHE STRING "Hugging Face bucket name (unused, kept for compatibility)")
set(HF_VERSION        "" CACHE STRING "Version to match for local archive (empty = resolve from git)")
set(HF_ENABLED        "" CACHE STRING "Whether to use prebuilt UI from local archives (ON/OFF)")
set(BUILD_UI          "" CACHE STRING "Build UI via npm (ON/OFF)")
set(LLAMA_UI_EMBED    "" CACHE STRING "Path to llama-ui-embed helper")
set(LLAMA_UI_GZIP     "" CACHE STRING "Apply gzip compress to assets to save bandwidth")

set(DIST_DIR     "${UI_BINARY_DIR}/dist")
set(SRC_DIST_DIR "${UI_SOURCE_DIR}/dist")
set(WORK_DIR     "${UI_BINARY_DIR}/ui-src")
set(STAMP_FILE   "${UI_BINARY_DIR}/.ui-stamp")
set(UI_CPP       "${UI_BINARY_DIR}/ui.cpp")
set(UI_H         "${UI_BINARY_DIR}/ui.h")

function(npm_build_should_skip out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    if(NOT EXISTS "${DIST_DIR}/index.html")
        return()
    endif()

    if(EXISTS "${STAMP_FILE}")
        return()
    endif()

    if(NOT EXISTS "${UI_SOURCE_DIR}/sources.cmake")
        return()
    endif()
    include("${UI_SOURCE_DIR}/sources.cmake")

    set(globs "")
    foreach(g ${UI_SOURCE_GLOBS})
        list(APPEND globs "${UI_SOURCE_DIR}/${g}")
    endforeach()
    file(GLOB_RECURSE sources ${globs})
    foreach(f ${UI_SOURCE_FILES})
        list(APPEND sources "${UI_SOURCE_DIR}/${f}")
    endforeach()

    file(TIMESTAMP "${DIST_DIR}/index.html" out_ts)

    foreach(s ${sources})
        if(NOT EXISTS "${s}")
            continue()
        endif()
        file(TIMESTAMP "${s}" s_ts)
        if(s_ts STRGREATER out_ts)
            return()
        endif()
    endforeach()

    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(stage_sources)
    if(EXISTS "${WORK_DIR}")
        file(GLOB staged RELATIVE "${WORK_DIR}" "${WORK_DIR}/*")
        list(REMOVE_ITEM staged "node_modules")
        foreach(entry ${staged})
            file(REMOVE_RECURSE "${WORK_DIR}/${entry}")
        endforeach()
    endif()

    file(COPY "${UI_SOURCE_DIR}/"
        DESTINATION "${WORK_DIR}"
        NO_SOURCE_PERMISSIONS
        PATTERN "node_modules" EXCLUDE
    )
endfunction()

function(npm_build out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    if(NOT EXISTS "${UI_SOURCE_DIR}/package.json")
        message(STATUS "UI: ${UI_SOURCE_DIR}/package.json not found, skipping build")
        return()
    endif()

    npm_build_should_skip(skip)
    if(skip)
        message(STATUS "UI: build output up-to-date, skipping")
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Prefer bun, fall back to npm
    find_program(BUN_EXECUTABLE NAMES bun bun.exe)
    if(BUN_EXECUTABLE)
        set(PKG_EXECUTABLE ${BUN_EXECUTABLE})
        message(STATUS "UI: using bun (${BUN_EXECUTABLE})")
    else()
        if(CMAKE_HOST_WIN32)
            find_program(NPM_EXECUTABLE NAMES npm.cmd npm.bat npm)
        else()
            find_program(NPM_EXECUTABLE npm)
        endif()
        if(NOT NPM_EXECUTABLE)
            message(STATUS "UI: neither bun nor npm found, skipping build")
            return()
        endif()
        set(PKG_EXECUTABLE ${NPM_EXECUTABLE})
        message(STATUS "UI: using npm (${NPM_EXECUTABLE})")
    endif()

    stage_sources()

    # Determine lockfile for staleness check (after staging copies sources)
    if(EXISTS "${WORK_DIR}/bun.lock")
        set(PKG_LOCKFILE "bun.lock")
    elseif(EXISTS "${WORK_DIR}/bun.lockb")
        set(PKG_LOCKFILE "bun.lockb")
    else()
        set(PKG_LOCKFILE "package-lock.json")
    endif()

    # Write our own marker after install so staleness works with either package manager
    set(DEPS_MARKER "${WORK_DIR}/node_modules/.ui-deps-stamp")
    set(need_install FALSE)
    if(NOT EXISTS "${DEPS_MARKER}")
        set(need_install TRUE)
    else()
        file(TIMESTAMP "${WORK_DIR}/${PKG_LOCKFILE}" lock_ts)
        file(TIMESTAMP "${DEPS_MARKER}" marker_ts)
        if(lock_ts STRGREATER marker_ts)
            set(need_install TRUE)
        endif()
    endif()

    if(need_install)
        message(STATUS "UI: running ${PKG_EXECUTABLE} install")
        execute_process(
            COMMAND ${PKG_EXECUTABLE} install
            WORKING_DIRECTORY "${WORK_DIR}"
            RESULT_VARIABLE rc
            ERROR_VARIABLE  err
        )
        if(NOT rc EQUAL 0)
            message(STATUS "UI: ${PKG_EXECUTABLE} install failed (${rc})")
            message(STATUS "  stderr: ${err}")
            return()
        endif()
        file(WRITE "${DEPS_MARKER}" "")
    endif()

    file(MAKE_DIRECTORY "${DIST_DIR}")

    message(STATUS "UI: running ${PKG_EXECUTABLE} run build, output -> ${DIST_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "LLAMA_UI_OUT_DIR=${DIST_DIR}" "LLAMA_UI_VERSION=${HF_VERSION}" "LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}"
                ${PKG_EXECUTABLE} run build
        WORKING_DIRECTORY "${WORK_DIR}"
        RESULT_VARIABLE rc
        ERROR_VARIABLE  err
    )
    if(NOT rc EQUAL 0)
        message(STATUS "UI: ${PKG_EXECUTABLE} run build failed (${rc})")
        message(STATUS "  stderr: ${err}")
        return()
    endif()

    if(NOT EXISTS "${DIST_DIR}/index.html")
        message(STATUS "UI: build finished but assets missing in ${DIST_DIR}")
        return()
    endif()

    message(STATUS "UI: build succeeded")
    file(REMOVE "${STAMP_FILE}")
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(resolve_version out_var)
    if(NOT "${HF_VERSION}" STREQUAL "")
        set(${out_var} "${HF_VERSION}" PARENT_SCOPE)
        return()
    endif()

    if(EXISTS "${LLAMA_SOURCE_DIR}/cmake/build-info.cmake")
        include("${LLAMA_SOURCE_DIR}/cmake/build-info.cmake")
        if(NOT "${BUILD_NUMBER}" STREQUAL "" AND NOT BUILD_NUMBER EQUAL 0)
            set(${out_var} "b${BUILD_NUMBER}" PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(local_archive_extract version out_var out_resolved)
    set(${out_var}      FALSE PARENT_SCOPE)
    set(${out_resolved} ""    PARENT_SCOPE)

    set(files_dir "${LLAMA_SOURCE_DIR}/files")
    if(NOT EXISTS "${files_dir}")
        message(STATUS "UI: local archive directory not found: ${files_dir}")
        return()
    endif()

    # Build candidate list: explicit version first, then any available archives (latest first)
    set(candidates "")
    if(NOT "${version}" STREQUAL "")
        list(APPEND candidates "${version}")
    endif()
    file(GLOB archives "${files_dir}/llama-b*-ui.tar.gz")
    if(archives)
        list(SORT archives ORDER DESCENDING)
        foreach(archive ${archives})
            get_filename_component(fname "${archive}" NAME_WE)
            string(REGEX REPLACE "llama-(b[0-9]+)-ui" "\\1" arch_ver "${fname}")
            list(APPEND candidates "${arch_ver}")
        endforeach()
    endif()

    foreach(resolved ${candidates})
        set(archive "${files_dir}/llama-${resolved}-ui.tar.gz")
        if(NOT EXISTS "${archive}")
            continue()
        endif()

        message(STATUS "UI: extracting local archive: ${archive}")

        file(REMOVE_RECURSE "${DIST_DIR}")
        file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${DIST_DIR}")

        # Flatten wrapper directory if the archive wraps assets in a top-level dir
        if(NOT EXISTS "${DIST_DIR}/index.html")
            file(GLOB wrapper_index "${DIST_DIR}/*/index.html")
            if(wrapper_index)
                list(GET wrapper_index 0 first_index)
                get_filename_component(wrapper_dir "${first_index}" DIRECTORY)
                message(STATUS "UI: flattening archive wrapper directory: ${wrapper_dir}")
                file(COPY "${wrapper_dir}/" DESTINATION "${DIST_DIR}")
                file(REMOVE_RECURSE "${wrapper_dir}")
            endif()
        endif()

        if(NOT EXISTS "${DIST_DIR}/index.html" OR NOT EXISTS "${DIST_DIR}/loading.html")
            message(STATUS "UI: archive ${archive} is missing required assets (index.html or loading.html)")
            continue()
        endif()

        message(STATUS "UI: local archive extracted successfully (${resolved})")
        set(${out_var}      TRUE          PARENT_SCOPE)
        set(${out_resolved} "${resolved}" PARENT_SCOPE)
        return()
    endforeach()

    if(NOT candidates)
        message(STATUS "UI: no local archives found in ${files_dir}")
    endif()
endfunction()

function(emit_files dist_dir)
    # If gzip is requested, compress every asset into a parallel _gzip/ tree
    # the structure stays the same; for ex: /abc/def --> /_gzip/abc/def
    # embed.cpp will check for _gzip and will pick it up
    if(LLAMA_UI_GZIP AND EXISTS "${dist_dir}/index.html")
        find_program(GZIP_EXECUTABLE gzip)
        if(NOT GZIP_EXECUTABLE)
            message(WARNING "UI: LLAMA_UI_GZIP requested but gzip not found, embedding uncompressed")
        else()
            set(gzip_dir "${dist_dir}/_gzip")
            file(REMOVE_RECURSE "${gzip_dir}")
            file(GLOB_RECURSE all_files RELATIVE "${dist_dir}" "${dist_dir}/*")
            foreach(f ${all_files})
                get_filename_component(dst_dir "${gzip_dir}/${f}" DIRECTORY)
                file(MAKE_DIRECTORY "${dst_dir}")
                execute_process(
                    COMMAND "${GZIP_EXECUTABLE}" -c "${dist_dir}/${f}"
                    OUTPUT_FILE "${gzip_dir}/${f}"
                    RESULT_VARIABLE gz_rc
                )
                if(NOT gz_rc EQUAL 0)
                    message(FATAL_ERROR "UI: gzip failed for ${f}")
                endif()
            endforeach()
            message(STATUS "UI: gzip compression applied (${gzip_dir})")
        endif()
    endif()

    set(args "${UI_CPP}" "${UI_H}")
    if(EXISTS "${dist_dir}/index.html")
        list(APPEND args "${dist_dir}")
    endif()

    execute_process(
        COMMAND "${LLAMA_UI_EMBED}" ${args}
        RESULT_VARIABLE rc
    )
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "UI: llama-ui-embed failed (${rc})")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# 1. Priority 1: pre-built assets supplied in tools/ui/dist
# ---------------------------------------------------------------------------
if(EXISTS "${SRC_DIST_DIR}/index.html")
    message(STATUS "UI: using pre-built assets from ${SRC_DIST_DIR}")
    emit_files("${SRC_DIST_DIR}")
    return()
endif()

# ---------------------------------------------------------------------------
# 2. Priority 2: npm build (if BUILD_UI=ON)
# ---------------------------------------------------------------------------
set(provisioned FALSE)

if(BUILD_UI)
    # Resolve version from git build-info if not explicitly set
    resolve_version(HF_VERSION)
    npm_build(NPM_OK)
    if(NPM_OK)
        set(provisioned TRUE)
    endif()
endif()

# ---------------------------------------------------------------------------
# 3. Priority 3: extract from local archive in ${LLAMA_SOURCE_DIR}/files
# ---------------------------------------------------------------------------
if(NOT provisioned AND HF_ENABLED)
    set(stamp_ok FALSE)
    set(stamped "")
    if(EXISTS "${STAMP_FILE}")
        file(READ "${STAMP_FILE}" stamped)
        string(STRIP "${stamped}" stamped)
        if(NOT "${stamped}" STREQUAL "")
            set(archive "${LLAMA_SOURCE_DIR}/files/llama-${stamped}-ui.tar.gz")
            if(EXISTS "${archive}" AND EXISTS "${DIST_DIR}/index.html" AND EXISTS "${DIST_DIR}/loading.html")
                set(stamp_ok TRUE)
            endif()
        endif()
    endif()

    if(stamp_ok)
        message(STATUS "UI: local archive '${stamped}' already extracted, skipping")
        set(provisioned TRUE)
    else()
        resolve_version(VERSION)
        local_archive_extract("${VERSION}" LOCAL_OK LOCAL_RESOLVED)
        if(LOCAL_OK)
            file(WRITE "${STAMP_FILE}" "${LOCAL_RESOLVED}")
            message(STATUS "UI: local archive extracted, stamp updated (${LOCAL_RESOLVED})")
            set(provisioned TRUE)
        else()
            message(STATUS "UI: local archive extraction failed")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# 4. Fallback: warn about stale or missing assets, then emit whatever we have
# ---------------------------------------------------------------------------
if(NOT provisioned)
    if(EXISTS "${DIST_DIR}/index.html")
        message(WARNING "UI: provisioning failed; embedding stale assets from ${DIST_DIR}")
    else()
        message(WARNING "UI: no assets available - building without an embedded UI. "
                        "Place a pre-built archive (llama-b<version>-ui.tar.gz) in the "
                        "'files' directory at the project root.")
    endif()
endif()

emit_files("${DIST_DIR}")
