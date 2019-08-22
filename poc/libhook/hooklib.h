#pragma once

typedef void (*autogen_cb_t)(void* shadow_base);
extern "C" void __libhook_register_init_cb(autogen_cb_t cb);

