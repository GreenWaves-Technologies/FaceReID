#include "StaticUserManager.h"
#include "face_db.h"

int initHandler(struct pi_device* fs)
{
    PRINTF("Loading static ReID database\n");
    int status = load_static_db(fs);
    if(!status)
    {
        PRINTF("Static DB load failed!\n");
    }

    return status;
}

int handleStranger(short* descriptor)
{
    (void) descriptor;
    return 0;
}
