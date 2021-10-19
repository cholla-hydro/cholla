#ifndef __NVTX_H__
#define __NVTX_H__

#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);


struct nvtx_raii {
  nvtx_raii(const char name[], int color_id) {
#ifdef USE_NVTX
    color_id = color_id%num_colors;
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = colors[color_id];
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
#endif
  }

  ~nvtx_raii() {
#ifdef USE_NVTX
    nvtxRangePop();
#endif
  }
};

#endif
