// Console functions

#pragma once

#include "common.h"

#include <functional>
#include <string>
#include <vector>

enum display_type {
    DISPLAY_TYPE_RESET = 0,
    DISPLAY_TYPE_INFO,
    DISPLAY_TYPE_PROMPT,
    DISPLAY_TYPE_REASONING,
    DISPLAY_TYPE_USER_INPUT,
    DISPLAY_TYPE_ERROR
};

namespace console {
    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_type display);
    bool readline(std::string & line, bool multiline_input);
    // 输入流是否已到 EOF。readline 返回空行既可能是用户敲了空行、也可能是流结束，
    // 会话循环必须用本接口区分，否则 EOF 时会无限重读（历史缺陷：每次 EOF 还会
    // 向进程组发一次 CTRL_C_EVENT，重定向输入时堆积数十万次直至栈溢出）。
    bool input_eof();

    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;
    void set_completion_callback(completion_callback cb);

    namespace spinner {
        void start();
        void stop();
    }

    // note: the logging API below output directly to stdout
    // it can negatively impact performance if used on inference thread
    // only use in in a dedicated CLI thread
    // for logging in inference thread, use log.h instead

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void log(const char * fmt, ...);

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void error(const char * fmt, ...);

    void flush();
}
