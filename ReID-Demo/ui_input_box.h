#ifndef __UI_INPUT_BOX_H__
#define __UI_INPUT_BOX_H__

#define INPUT_BOX_MAX_LENGTH 16

// Returns input text (after ENTER) or NULL in case of cancel (ESC)
char* input_box(struct pi_device *display, int top, char* caption);

#endif